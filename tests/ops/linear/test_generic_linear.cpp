// The generic projection path (ops/generic): every weight format at random and awkward shapes
// (N and K off the multiples of 64/128, small N and K, K the register MMA cannot tile), through
// the public linear / linear_add / linear_swiglu entry points and through every compiled route,
// against an FP64 reference GEMM of the exactly decoded weight. Then the route contract: every
// route the prefill profile may select at T >= 17 gives a token the same bytes whatever the call's
// size, start or route. W4A8 (AllowA8, int8 activations with a scale per 32 values) against the
// same oracle under the A8 criterion, per format, entry point and compiled schedule.
#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/device_class.h"
#include "core/weight.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/linear_add.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/common/native_nvfp4.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/r8_decode_table.h"
#include "ops/linear/linear_registry.h"
#include "ops/linear_add/q8/q8_linear_add_plan.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_plan.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <deque>
#include <memory>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::test;
using namespace ninferno::test::linear;
namespace generic = ninferno::ops::detail::generic;

struct HostWeight {
    quantized_weight::PackedWeight packed; // unused for BF16
    std::vector<std::uint16_t> bf16;       // BF16 only
    QType qtype;
    std::int32_t n, k;
};

struct Format {
    const char* name;
    QType qtype;
    int n_multiple; // layout requirement
    int k_multiple;
};

// check_mmq_code_sums: ggml's matrix kernel against the FP64 oracle of the int8 activation, relative
// L2. RTX 5070 Ti: 0.0024 with the block sums from the codes (Q4_K and Q5_K alike), 0.056 with the
// raw values' sum.
constexpr double kMmqCodeSumAgreement = 0.01;

const Format kFormats[] = {
    {"q4", QType::Q4_G64_FP16, 1, 1},
    {"q5", QType::Q5_G64_FP16, 1, 1},
    {"q6", QType::Q6_G64_FP16, 1, 1},
    {"q8", QType::Q8_G32_FP16, 1, 1},
    {"q3g128", QType::Q3_G128_FP16, 1, 1},
    {"q3g64", QType::Q3_G64_FP16, 1, 1},
    {"fp8", QType::FP8_E4M3FN_ROW_BF16, 1, 1},
    {"nvfp4", QType::NVFP4, 1, 64},
    {"bf16", QType::BF16, 1, 1},
    {"mxfp4", QType::MXFP4, 1, 32},
    {"fp8r32", QType::FP8_E4M3FN_ROW_F32, 1, 1},
    {"fp8b128", QType::FP8_E4M3FN_B128_F32, 1, 1},
    {"u4zp", QType::U4_G128_ZP_FP16, 1, 128},
    {"u4zp32", QType::U4_G32_ZP_FP16, 1, 32},
    {"u4zp64", QType::U4_G64_ZP_FP16, 1, 64},
    {"q4k", QType::GGML_Q4_K, 1, 256},
    {"q5k", QType::GGML_Q5_K, 1, 256},
    {"q6k", QType::GGML_Q6_K, 1, 256},
    {"q3k", QType::GGML_Q3_K, 1, 256},
    {"q2k", QType::GGML_Q2_K, 1, 256},
    {"iq4xs", QType::GGML_IQ4_XS, 1, 256},
    {"iq3xxs", QType::GGML_IQ3_XXS, 1, 256},
    {"iq3s", QType::GGML_IQ3_S, 1, 256},
    {"q40", QType::GGML_Q4_0, 1, 32},
    {"q41", QType::GGML_Q4_1, 1, 32},
    {"q50", QType::GGML_Q5_0, 1, 32},
    {"q51", QType::GGML_Q5_1, 1, 32},
    {"iq4nl", QType::GGML_IQ4_NL, 1, 32},
    {"iq2xxs", QType::GGML_IQ2_XXS, 1, 256},
    {"iq2xs", QType::GGML_IQ2_XS, 1, 256},
    {"iq2s", QType::GGML_IQ2_S, 1, 256},
    {"iq1s", QType::GGML_IQ1_S, 1, 256},
    {"iq1m", QType::GGML_IQ1_M, 1, 256},
    {"tq10", QType::GGML_TQ1_0, 1, 256},
    {"tq20", QType::GGML_TQ2_0, 1, 256},
    {"q10", QType::GGML_Q1_0, 1, 128},
    {"q20", QType::GGML_Q2_0, 1, 64},
};

// The test's entry for `qtype` (sections that need one format look it up, not its position).
const Format& format_of(QType qtype) {
    for (const Format& f : kFormats) {
        if (f.qtype == qtype) return f;
    }
    throw std::logic_error("generic linear test: no format entry for this qtype");
}

// NVFP4 weights run on every build (the register-MMA and scalar routes); the W4A4 and tiled NVFP4
// routes only where the native NVFP4 kernels are built.
constexpr bool kNativeNvfp4 = ops::detail::kNativeNvfp4;
// NINFERNO_TEST_FORMATS=iq3xxs,q6k,...: only those formats' cases (a quick check of the formats a
// change touches); unset, every format.
bool selected(const Format& f) {
    static const std::string list = [] {
        const char* v = std::getenv("NINFERNO_TEST_FORMATS");
        return v == nullptr ? std::string() : "," + std::string(v) + ",";
    }();
    return list.empty() || list.find("," + std::string(f.name) + ",") != std::string::npos;
}

bool on_build(const Format& f) { return selected(f); }

bool ggml_kquant(QType q) { return is_ggml_block(q); }
// ggml's 32-value block formats (Q4_0 .. IQ4_NL).
bool ggml_block32(QType q) { return is_ggml_block(q) && ggml_block_values(q) == 32; }

// One awkward K per format for the route sweeps: not a multiple of 64 where the layout allows.
std::int32_t awkward_k(const Format& f) {
    return f.qtype == QType::NVFP4             ? 1088
           : f.qtype == QType::MXFP4           ? 1056
           : f.qtype == QType::U4_G128_ZP_FP16 ? 1152
           : f.qtype == QType::U4_G64_ZP_FP16  ? 1088
           : f.qtype == QType::U4_G32_ZP_FP16  ? 1088
           : ggml_block32(f.qtype)             ? 1152
           : ggml_kquant(f.qtype)              ? 1280
                                               : 1040;
}

HostWeight make_weight(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed) {
    HostWeight w{{}, {}, qtype, n, k};
    switch (qtype) {
    case QType::Q4_G64_FP16:
        w.packed = make_q4_g64_fp16_weight(n, k, seed);
        break;
    case QType::Q5_G64_FP16:
        w.packed = make_q5_g64_fp16_weight(n, k, seed);
        break;
    case QType::Q6_G64_FP16:
        w.packed = make_q6_g64_fp16_weight(n, k, seed);
        break;
    case QType::Q8_G32_FP16:
        w.packed = make_q8_g32_fp16_weight(n, k, seed);
        break;
    case QType::Q3_G128_FP16:
    case QType::Q3_G64_FP16:
        w.packed = quantized_weight::make_patterned_weight(qtype, n, k, seed);
        break;
    case QType::FP8_E4M3FN_ROW_BF16:
        w.packed = make_fp8_weight(n, k, seed);
        break;
    case QType::NVFP4:
        w.packed = make_nvfp4_weight(n, k, seed);
        break;
    case QType::MXFP4:
    case QType::FP8_E4M3FN_ROW_F32:
    case QType::FP8_E4M3FN_B128_F32:
    case QType::U4_G128_ZP_FP16:
    case QType::U4_G32_ZP_FP16:
    case QType::U4_G64_ZP_FP16:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_Q3_K:
    case QType::GGML_Q2_K:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_Q4_0:
    case QType::GGML_Q4_1:
    case QType::GGML_Q5_0:
    case QType::GGML_Q5_1:
    case QType::GGML_IQ4_NL:
    case QType::GGML_IQ2_XXS:
    case QType::GGML_IQ2_XS:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ1_S:
    case QType::GGML_IQ1_M:
    case QType::GGML_TQ1_0:
    case QType::GGML_TQ2_0:
    case QType::GGML_Q1_0:
    case QType::GGML_Q2_0:
        w.packed = quantized_weight::make_patterned_weight(qtype, n, k, seed);
        break;
    case QType::BF16: {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0F, 0.05F);
        w.bf16.resize(static_cast<std::size_t>(n) * k);
        for (auto& v : w.bf16) v = f32_to_bf16(normal(rng));
        break;
    }
    default:
        throw std::invalid_argument("unsupported format");
    }
    return w;
}

std::vector<float> weight_rows(const HostWeight& w, const std::vector<std::int32_t>& rows) {
    if (w.qtype != QType::BF16) return quantized_weight::materialize_rows_fp32(w.packed, rows);
    std::vector<float> out(rows.size() * static_cast<std::size_t>(w.k));
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::int32_t c = 0; c < w.k; ++c)
            out[i * w.k + c] = bf16_to_f32(w.bf16[static_cast<std::size_t>(rows[i]) * w.k + c]);
    return out;
}

struct DeviceWeight {
    DeviceBuffer buffer;
    Weight weight;
};

DeviceWeight upload_weight(const HostWeight& w) {
    if (w.qtype == QType::BF16) {
        DeviceBuffer b(w.bf16.size() * 2);
        b.copy_from_host(w.bf16.data(), b.bytes);
        Weight wt{};
        wt.qtype  = QType::BF16;
        wt.layout = QuantLayout::Contiguous;
        wt.qdata  = b.p;
        wt.payload = b.p;
        wt.payload_bytes = b.bytes;
        wt.n = wt.shape[0] = wt.padded_shape[0] = w.n;
        wt.k = wt.shape[1] = wt.padded_shape[1] = w.k;
        wt.ndim = 2;
        return {std::move(b), wt};
    }
    DeviceBuffer b(w.packed.payload.size());
    b.copy_from_host(w.packed.payload.data(), b.bytes);
    const Weight wt = w.packed.device_weight(b.p);
    return {std::move(b), wt};
}

std::vector<std::int32_t> pick_rows(std::int32_t extent, std::int32_t limit) {
    std::vector<std::int32_t> rows;
    if (extent <= limit) {
        for (std::int32_t i = 0; i < extent; ++i) rows.push_back(i);
        return rows;
    }
    for (std::int32_t i = 0; i < limit; ++i) {
        rows.push_back(static_cast<std::int32_t>((static_cast<std::int64_t>(extent - 1) * i) / (limit - 1)));
    }
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

std::vector<std::uint16_t> download_bf16(const void* p, std::size_t count) {
    std::vector<std::uint16_t> out(count);
    cuda_check(cudaMemcpy(out.data(), p, count * 2, cudaMemcpyDeviceToHost), "download");
    return out;
}

// The W4A8 route's activation quantization (generic_w4a8.cu), in the same FP32 operations: per
// token and 32 values, s = amax/127 and q = rint(x * (127/amax)) clamped to [-127, 127]; x_q = q*s.
// The I8 route's (generic_i8.cu) is the same per 64 values.
void quantize_a8(std::vector<float>& x, std::int32_t k, int group = 32) {
    const auto g = static_cast<std::size_t>(group);
    for (std::size_t base = 0; base + g <= x.size(); base += g) {
        if ((base % static_cast<std::size_t>(k)) + g > static_cast<std::size_t>(k)) continue;
        float amax = 0.0F;
        for (std::size_t i = 0; i < g; ++i) amax = std::max(amax, std::fabs(x[base + i]));
        const float inv   = amax > 0.0F ? 127.0F / amax : 0.0F;
        const float scale = amax / 127.0F;
        for (std::size_t i = 0; i < g; ++i) {
            const float q = std::min(127.0F, std::max(-127.0F, std::nearbyint(x[base + i] * inv)));
            x[base + i]   = q * scale;
        }
    }
}

// ggml's q8_1 activation error enters the projections (gate and up, or relu's v), and the epilogue
// carries it to f with the sensitivity |g df/dg| + |u df/du|, not with |f|: where GELU flattens, a
// gate near -5 gives f near 1e-5 while the gate keeps its projection's error, and that error
// relative to f is unbounded. The criterion is the route's A8 allowance with that sensitivity as
// the reference scale (for relu^2 it is 2|f|; for a product near its regular range, 2..3 |f|).
int verify_propagated(std::string_view label, const std::vector<double>& got,
                      const std::vector<double>& ref, const std::vector<double>& scale,
                      const ReductionCriterion& criterion) {
    double squared_error = 0.0, squared_scale = 0.0, largest_error = 0.0, largest_scale = 0.0;
    std::size_t at = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) {
            std::cerr << label << ": non-finite value at index " << i << '\n';
            return 1;
        }
        const double error = std::fabs(got[i] - ref[i]);
        squared_error += error * error;
        squared_scale += scale[i] * scale[i];
        largest_scale = std::max(largest_scale, scale[i]);
        if (error > largest_error) {
            largest_error = error;
            at            = i;
        }
    }
    const double relative_l2 = std::sqrt(squared_error) / std::max(std::sqrt(squared_scale), 1e-30);
    const double gross_limit =
        criterion.gross_absolute + criterion.gross_relative_to_max_reference * largest_scale;
    if (error_stats_enabled()) {
        std::cout << "PROPAGATED_STATS rel_l2_ratio=" << relative_l2 / criterion.relative_l2
                  << " gross_ratio=" << largest_error / gross_limit << " case=" << label << '\n';
    }
    if (relative_l2 <= criterion.relative_l2 && largest_error <= gross_limit) return 0;
    std::cerr << label << ": propagated criterion failed at index " << at << " actual=" << got[at]
              << " reference=" << ref[at] << " relative_l2=" << relative_l2 << " of "
              << criterion.relative_l2 << " max_error=" << largest_error << " of " << gross_limit
              << '\n';
    return 1;
}

// The last few weights check_case built, by (format, rows, K, seed): one shape's calls at every
// T and op share its weight instead of quantizing, uploading and freeing it per call (host
// quantization and cudaMalloc/cudaFree were most of the test's 126 s). A call copies the
// Weight view, so an ordered call's layout and k_order stay its own.
struct CachedWeight {
    QType qtype;
    std::int32_t rows, k;
    std::uint32_t seed;
    std::shared_ptr<const HostWeight> host;
    std::shared_ptr<const DeviceWeight> device;
};

const CachedWeight& cached_weight(QType qtype, std::int32_t rows, std::int32_t k,
                                  std::uint32_t seed) {
    // Never destroyed: its device buffers must not be freed after the CUDA runtime shut down.
    static auto& cache = *new std::deque<CachedWeight>;
    for (const auto& c : cache)
        if (c.qtype == qtype && c.rows == rows && c.k == k && c.seed == seed) return c;
    auto host   = std::make_shared<const HostWeight>(make_weight(qtype, rows, k, seed));
    auto device = std::make_shared<const DeviceWeight>(upload_weight(*host));
    if (cache.size() == 4) cache.pop_front();
    cache.push_back({qtype, rows, k, seed, std::move(host), std::move(device)});
    return cache.back();
}

enum class Op { Linear, Add, SwiGlu, GeGlu, Relu2 };

const char* op_name(Op op) {
    return op == Op::Linear   ? "linear"
           : op == Op::Add    ? "add"
           : op == Op::SwiGlu ? "swiglu"
           : op == Op::GeGlu  ? "geglu"
                              : "relu2";
}

// SwiGLU and GeGLU: a gate/up weight of twice the output rows, the SwiGlu epilogue.
bool gated(Op op) { return op == Op::SwiGlu || op == Op::GeGlu; }

// Runs one call (public entry point, or an explicit route when `route` is set) and compares the
// selected output rows of every token with the FP64 oracle. `ordered`: the weight's K values are
// stored in a random column order (RowGroupZpOrdered, act-order), so the op must gather its input
// x[order[j]] for stored column j.
int check_case(const Format& f, Op op, std::int32_t out_rows, std::int32_t k, std::int32_t t,
               ops::LinearPolicy policy, std::uint32_t seed, const generic::Route* route,
               ActivationCompute compute, bool ordered = false) {
    const std::int32_t weight_rows_n = gated(op) ? 2 * out_rows : out_rows;
    const CachedWeight& cached       = cached_weight(f.qtype, weight_rows_n, k, seed);
    const HostWeight& host           = *cached.host;
    Weight weight                    = cached.device->weight;
    std::vector<std::int32_t> order;
    DeviceBuffer order_buffer(ordered ? static_cast<std::size_t>(k) * 4 : 4);
    if (ordered) {
        order.resize(static_cast<std::size_t>(k));
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), std::mt19937(seed + 13U));
        order_buffer.copy_from_host(order.data(), order_buffer.bytes);
        weight.layout  = QuantLayout::RowGroupZpOrdered;
        weight.k_order = order_buffer.p;
    }
    // Without tensor mma.sync (Volta) the activation route of an AllowA8 call follows the device:
    // from 17 tokens a tiled format takes the tiled tensor-core route with f16 activations
    // (volta_tiled_route). A pinned route keeps the compute its caller states.
    if (route == nullptr && policy == ops::LinearPolicy::AllowA8 &&
        compute == ActivationCompute::A8 && generic::volta_tiled_route(weight, policy, t)) {
        compute = ActivationCompute::A16;
    }
    // The A4 criterion is stated for the hashed activation pattern the A4 suites use.
    const auto activation = make_linear_activation(k, t, seed + 7U, compute);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(out_rows) * t);
    {
        std::mt19937 rng(seed + 11U);
        std::uniform_real_distribution<float> u(-0.5F, 0.5F);
        for (auto& v : residual) v = f32_to_bf16(u(rng));
    }
    DeviceBuffer out(residual.size() * 2);
    if (op == Op::Add) {
        out.copy_from_host(residual.data(), out.bytes);
    } else {
        cuda_check(cudaMemset(out.p, 0xff, out.bytes), "poison");
    }
    Tensor tx(x.p, DType::BF16, {k, t});
    Tensor tout(out.p, DType::BF16, {out_rows, t});

    std::size_t capacity = 0;
    if (op == Op::Linear || op == Op::Relu2)
        capacity = ops::linear_workspace_capacity_bytes(f.qtype, out_rows, k, policy, t, t);
    else if (op == Op::Add)
        capacity = ops::linear_add_workspace_capacity_bytes(f.qtype, out_rows, k, policy, t, t);
    else if (op == Op::SwiGlu)
        capacity = ops::linear_swiglu_workspace_capacity_bytes(f.qtype, 2 * out_rows, k, policy, t, t);
    else
        capacity =
            ops::linear_geglu_workspace_capacity_bytes(f.qtype, 2 * out_rows, k, policy, t, t);
    if (route != nullptr) capacity = std::max(capacity, generic::route_workspace_bytes(*route, k, t));
    if (route != nullptr && route->family == generic::Family::GgmlMmq) {
        capacity = std::max(capacity, generic::ggml_mmq_workspace_bytes(k, weight_rows_n, t));
    }
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));

    const std::string label = std::string(f.name) + " " + op_name(op) + " [" +
                              std::to_string(weight_rows_n) + "," + std::to_string(k) +
                              "] T=" + std::to_string(t) +
                              (route ? " route=" + generic::route_name(*route) : std::string()) +
                              (policy == ops::LinearPolicy::AllowA4   ? " A4"
                               : policy == ops::LinearPolicy::AllowA8 ? " A8"
                                                                      : "") +
                              (ordered ? " ordered" : "");
    try {
        if (route != nullptr) {
            generic::Call call;
            call.x               = &tx;
            call.weight          = &weight;
            call.rows            = out_rows;
            call.up_offset       = gated(op) ? out_rows : 0;
            call.gelu_gate       = op == Op::GeGlu;
            call.relu2           = op == Op::Relu2;
            call.epilogue        = (op == Op::Linear || op == Op::Relu2) ? generic::Epilogue::Store
                                   : op == Op::Add ? generic::Epilogue::AddResidual
                                                   : generic::Epilogue::SwiGlu;
            call.out[0]          = &tout;
            call.segment_rows[0] = out_rows;
            const std::uint64_t fallbacks = generic::ggml_mmq_fallback_count();
            const std::uint64_t demotions = generic::route_demotion_count();
            generic::run_route(*route, call, &workspace, nullptr);
            if (route->family == generic::Family::GgmlMmq &&
                generic::ggml_mmq_fallback_count() != fallbacks) {
                std::cerr << label << ": ggml_mmq ran A16 in a workspace sized for it\n";
                return 1;
            }
            // The route asked for ran (or one of the same bytes): a call sized and aligned for it
            // is never demoted.
            if (generic::route_demotion_count() != demotions) {
                std::cerr << label << ": " << generic::route_name(*route)
                          << " ran another route in a call sized for it\n";
                return 1;
            }
        } else if (op == Op::Linear) {
            ops::linear(tx, weight, tout, policy, workspace, nullptr);
        } else if (op == Op::Relu2) {
            generic::linear_relu2(tx, weight, tout, policy, &workspace, nullptr);
        } else if (op == Op::Add) {
            ops::linear_add(tx, weight, tout, policy, workspace, nullptr);
        } else if (op == Op::SwiGlu) {
            ops::linear_swiglu(tx, weight, tout, policy, workspace, nullptr);
        } else {
            ops::linear_geglu(tx, weight, tout, policy, workspace, nullptr);
        }
        cuda_check(cudaDeviceSynchronize(), "run");
    } catch (const std::exception& e) {
        std::cerr << label << ": unexpected exception: " << e.what() << '\n';
        return 1;
    }
    if (workspace.peak_used() > capacity) {
        std::cerr << label << ": workspace capacity " << capacity << " below use "
                  << workspace.peak_used() << '\n';
        return 1;
    }

    const auto rows = pick_rows(out_rows, 40);
    std::vector<std::int32_t> source_rows = rows;
    if (gated(op))
        for (auto r : rows) source_rows.push_back(r + out_rows);
    const auto w = weight_rows(host, source_rows);
    std::vector<float> xa(activation.size());
    for (std::size_t i = 0; i < activation.size(); ++i) xa[i] = bf16_to_f32(activation[i]);
    if (ordered) {
        // Stored column j multiplies input column order[j].
        const std::vector<float> natural = xa;
        for (std::int32_t token = 0; token < t; ++token)
            for (std::int32_t j = 0; j < k; ++j)
                xa[static_cast<std::size_t>(token) * k + j] =
                    natural[static_cast<std::size_t>(token) * k + order[static_cast<std::size_t>(j)]];
    }
    // The W4A8 route computes sum(w * x_q) up to FP32 accumulation, x_q its int8 quantization per
    // 32 values: the oracle takes x_q, emulated bit for bit, and the route answers under the A16
    // criterion (the quantization error itself is not the kernel's to bound).
    // ggml's matrix kernel quantizes to its own q8_1 (a K-quant's min term from the sum of the
    // unquantized values, Q2_K a scale per 64): the oracle keeps x and the route answers under the
    // A8 allowance.
    // So does ggml's vector kernel, which the dispatch itself takes at verify widths (route null).
    const generic::Problem problem{f.qtype, policy, out_rows, k,
                                   op == Op::Add ? generic::Epilogue::AddResidual
                                   : gated(op)   ? generic::Epilogue::SwiGlu
                                                 : generic::Epilogue::Store};
    const generic::Route taken =
        route != nullptr ? *route : generic::select_route(problem, weight, t, false);
    const bool mmq_route =
        taken.family == generic::Family::GgmlMmq || taken.family == generic::Family::GgmlVec;
    // The dispatch's route decides what it computes in: where ggml's kernels take a call the
    // section expected A16 of (the R8 and W4A8 rules it models), x is quantized all the same.
    if (mmq_route && route == nullptr && compute == ActivationCompute::A16 &&
        policy != ops::LinearPolicy::A16Only) {
        compute = ActivationCompute::A8;
    }
    // ... and where the cost model weighs an A16 tile faster than the int8 ones the section
    // expected (AllowA8 permits int8, it does not require it), x stays BF16.
    const bool a8_route = mmq_route || taken.family == generic::Family::W4a8Mma ||
                          taken.family == generic::Family::R8Mma ||
                          taken.family == generic::Family::I8Mma;
    if (route == nullptr && compute == ActivationCompute::A8 && !a8_route) {
        compute = ActivationCompute::A16;
    }
    // A product of projections (gate and up, relu^2) on ggml's q8_1 kernels compounds the
    // activation error past the A8 allowance of x at long K (q4k K 12288 T 2: 1.6 % of x, 0.28 % of
    // x's int8), while formats with a min term (q4_1, q5_1, K-quants) take that term from the sum
    // of the unquantized x, which the int8 oracle misses: those ops pass against either oracle.
    const bool product_on_q8 =
        compute == ActivationCompute::A8 && mmq_route && (gated(op) || op == Op::Relu2);
    std::vector<float> xq;
    if (product_on_q8) {
        xq = xa;
        quantize_a8(xq, k);
    }
    if (compute == ActivationCompute::A8 && !mmq_route) {
        quantize_a8(xa, k, taken.family == generic::Family::I8Mma ? 64 : 32);
    }
    std::vector<double> dots(source_rows.size() * static_cast<std::size_t>(t));
    cpu_linear_gemm_fp64(w.data(), xa.data(), dots.data(),
                         static_cast<std::int32_t>(source_rows.size()), k, t);
    std::vector<double> dots_q;
    if (product_on_q8) {
        dots_q.resize(dots.size());
        cpu_linear_gemm_fp64(w.data(), xq.data(), dots_q.data(),
                             static_cast<std::int32_t>(source_rows.size()), k, t);
    }
    const auto got_bits = download_bf16(out.p, residual.size());
    std::vector<double> got, ref, ref_q, scale;
    const auto epilogue = [&](const std::vector<double>& d, std::int32_t token, std::size_t i) {
        const std::size_t n_sel = source_rows.size();
        double value            = d[static_cast<std::size_t>(token) * n_sel + i];
        if (op == Op::Relu2) value = value > 0.0 ? value * value : 0.0;
        if (gated(op)) {
            const double up = d[static_cast<std::size_t>(token) * n_sel + rows.size() + i];
            value           = op == Op::GeGlu
                                  ? 0.5 * value *
                              (1.0 + std::tanh(0.79788456080286535588 *
                                                         (value + 0.044715 * value * value * value))) *
                              up
                                  : value / (1.0 + std::exp(-value)) * up;
        }
        return value;
    };
    if (product_on_q8) {
        for (std::int32_t token = 0; token < t; ++token) {
            for (std::size_t i = 0; i < rows.size(); ++i)
                ref_q.push_back(epilogue(dots_q, token, i));
        }
    }
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const std::size_t n_sel = source_rows.size();
            double value            = dots[static_cast<std::size_t>(token) * n_sel + i];
            if (gated(op)) {
                const double g  = value;
                const double up = dots[static_cast<std::size_t>(token) * n_sel + rows.size() + i];
                double act = 0.0, slope = 0.0;
                if (op == Op::GeGlu) {
                    constexpr double c = 0.79788456080286535588;
                    const double th    = std::tanh(c * (g + 0.044715 * g * g * g));
                    act                = 0.5 * g * (1.0 + th);
                    slope =
                        0.5 * (1.0 + th) + 0.5 * g * (1.0 - th * th) * c * (1.0 + 0.134145 * g * g);
                } else {
                    const double sigma = 1.0 / (1.0 + std::exp(-g));
                    act                = g * sigma;
                    slope              = sigma + g * sigma * (1.0 - sigma);
                }
                scale.push_back(std::fabs(act * up) + std::fabs(g * slope * up));
            } else if (op == Op::Relu2) {
                scale.push_back(value > 0.0 ? 2.0 * value * value : 0.0);
            }
            if (op == Op::Relu2) value = value > 0.0 ? value * value : 0.0;
            if (op == Op::Add) value += bf16_to_f32(residual[static_cast<std::size_t>(token) * out_rows + rows[i]]);
            if (gated(op)) {
                const double up = dots[static_cast<std::size_t>(token) * n_sel + rows.size() + i];
                value           = op == Op::GeGlu
                                      ? 0.5 * value *
                                  (1.0 + std::tanh(0.79788456080286535588 *
                                                             (value + 0.044715 * value * value * value))) *
                                  up
                                      : value / (1.0 + std::exp(-value)) * up;
            }
            ref.push_back(value);
            got.push_back(bf16_to_f32(got_bits[static_cast<std::size_t>(token) * out_rows + rows[i]]));
        }
    }
    // Poison check: every output element written.
    for (const auto bits : got_bits) {
        if (bits == 0xffffU) {
            std::cerr << label << ": output retains poison\n";
            return 1;
        }
    }
    ReductionCriterion criterion = linear_criterion(
        compute == ActivationCompute::A8 && !mmq_route ? ActivationCompute::A16 : compute);
    if (gated(op) &&
        (compute == ActivationCompute::A4 || (compute == ActivationCompute::A8 && mmq_route))) {
        // Two quantized projections feed the product (A4, or ggml's q8_1 kernels at A8), so the
        // gross tail is twice Linear's; the relative-L2 allowance is Linear's, as in the
        // LinearSwiGLU suite.
        criterion.gross_relative_to_max_reference *= 2.0;
    }
    if (op == Op::Relu2 &&
        (compute == ActivationCompute::A4 || (compute == ActivationCompute::A8 && mmq_route))) {
        // relu(v)^2 doubles the relative error of v (d(v^2) = 2 v dv): the activation quantization
        // error of the projection (A4, or ggml's q8_1 kernels at A8) is Linear's, squared through
        // the epilogue.
        criterion.relative_l2 *= 2.0;
        criterion.gross_relative_to_max_reference *= 2.0;
    }
    if (mmq_route && op == Op::Linear) {
        // Diagnostic: relative L2 to the oracle of the unquantized x and to that of W4A8's q8.
        std::vector<float> xq = xa;
        quantize_a8(xq, k);
        std::vector<double> dq(dots.size());
        cpu_linear_gemm_fp64(w.data(), xq.data(), dq.data(),
                             static_cast<std::int32_t>(source_rows.size()), k, t);
        double ex = 0.0, eq = 0.0, rr = 0.0;
        for (std::int32_t token = 0; token < t; ++token) {
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const std::size_t at = static_cast<std::size_t>(token) * rows.size() + i;
                const std::size_t d  = static_cast<std::size_t>(token) * source_rows.size() + i;
                ex += (got[at] - dots[d]) * (got[at] - dots[d]);
                eq += (got[at] - dq[d]) * (got[at] - dq[d]);
                rr += dots[d] * dots[d];
            }
        }
        std::cout << label << " mmq rel_l2 to x " << std::sqrt(ex / rr) << ", to W4A8's q8 "
                  << std::sqrt(eq / rr) << "\n";
    }
    if (product_on_q8) {
        const ReductionCriterion a8 = linear_criterion(ActivationCompute::A8);
        const ReductionStats q      = compute_reduction_stats(got.data(), ref_q.data(),
                                                              static_cast<std::int64_t>(got.size()));
        if (reduction_passes(q, static_cast<std::int64_t>(got.size()), a8)) {
            return verify_reduction(label + " (int8 x)", got, ref_q, a8);
        }
    }
    if (mmq_route && (gated(op) || op == Op::Relu2)) {
        return verify_propagated(label, got, ref, scale, criterion);
    }
    return verify_reduction(label, got, ref, criterion);
}

bool tiled_format(QType qtype) {
    Weight probe;
    probe.qtype = qtype;
    return generic::tiled_supports(generic::codec_of(probe));
}

// Whether the register-MMA schedule is compiled for the format at K (not a 2 x 8 tile of a
// decoder that reads 32-value blocks: u4 G32/G64 take one at K % 128 == 0).
bool rmma_compiled(QType qtype, int schedule, std::int32_t k = 0) {
    Weight probe;
    probe.qtype  = qtype;
    const auto s = generic::rmma_schedule_info(schedule);
    return generic::rmma_schedule_launchable(schedule) &&
           generic::rmma_tile_compiled(generic::codec_of(probe), s.m_tiles, s.n_tiles, k);
}

// Every T >= 17 column must be byte-identical across routes admitted to the prefill profile and
// across call sizes and starts. Op::Add adds into a residual that every call starts from (the
// window's columns of one residual).
int check_invariance(const Format& f, Op op, std::int32_t out_rows, std::int32_t k,
                     ops::LinearPolicy policy, std::uint32_t seed,
                     const std::vector<generic::Route>* only = nullptr) {
    const std::int32_t wide  = 300;
    const std::int32_t rows_n = gated(op) ? 2 * out_rows : out_rows;
    const HostWeight host    = make_weight(f.qtype, rows_n, k, seed);
    DeviceWeight device      = upload_weight(host);
    const auto activation    = make_linear_activation(k, wide, seed + 3U, ActivationCompute::A4);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    const auto residual = make_linear_activation(out_rows, wide, seed + 5U, ActivationCompute::A16);
    DeviceArena workspace(std::max<std::size_t>(
        std::max(generic::route_workspace_bytes(generic::Route{generic::Family::W4a4Mma, 0}, k, wide),
                 std::max(generic::route_workspace_bytes(generic::Route{generic::Family::W4a8Mma, 0}, k, wide),
                          generic::route_workspace_bytes(generic::Route{generic::Family::I8Mma, 0}, k, wide))),
        256));
    const bool a4 = f.qtype == QType::NVFP4 && policy == ops::LinearPolicy::AllowA4;
    const bool a8 = generic::w4a8_supports(f.qtype) && policy == ops::LinearPolicy::AllowA8;

    std::vector<generic::Route> routes;
    if (only != nullptr) {
        routes = *only;
    } else if (a4) {
        for (int i = 0; i < generic::w4a4_schedule_count(); ++i)
            routes.push_back({generic::Family::W4a4Mma, static_cast<std::uint8_t>(i)});
    } else if (a8) {
        // The schedules the weight is admitted to (the producer-warp tile: five formats).
        for (int i = 0; i < generic::w4a8_schedule_count(); ++i) {
            const generic::Route r{generic::Family::W4a8Mma, static_cast<std::uint8_t>(i)};
            if (generic::route_admits(r, device.weight, policy)) routes.push_back(r);
        }
    } else if (tiled_format(f.qtype) && (k % 64) == 0) {
        // A format with a tiled codec takes only tiled routes under the prefill profile (whole
        // 64-value K tiles; q4/q5/q6_g64 at an awkward K stay on the register MMA).
        for (int i = 0; i < generic::tiled_schedule_count(); ++i) {
            if (!generic::tiled_fits(i, generic::codec_of(device.weight))) continue;
            routes.push_back({generic::Family::TiledMma, static_cast<std::uint8_t>(i)});
        }
    } else {
        for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
            const generic::Route r{generic::Family::RegisterMma, static_cast<std::uint8_t>(i)};
            if (generic::route_is_token_invariant(r) && rmma_compiled(f.qtype, i, k)) {
                routes.push_back(r);
            }
        }
        // The staged register MMA sums in the WarpsK == 1 order: the same bytes as those routes.
        for (int i = 0; i < generic::staged_schedule_count(); ++i) {
            const generic::Route r{generic::Family::StagedMma, static_cast<std::uint8_t>(i)};
            if (generic::route_admits(r, device.weight, policy)) routes.push_back(r);
        }
        if (f.qtype == QType::NVFP4) {
            for (int i = 0; kNativeNvfp4 && i < generic::nvfp4_tiled_schedule_count(); ++i) {
                const generic::Route r{generic::Family::Nvfp4TiledMma, static_cast<std::uint8_t>(i)};
                if (generic::route_is_token_invariant(r)) routes.push_back(r);
            }
        }
    }
    const auto run = [&](generic::Route route, std::int32_t offset, std::int32_t t,
                         std::vector<std::uint16_t>& bits) {
        DeviceBuffer out(static_cast<std::size_t>(out_rows) * t * 2);
        if (op == Op::Add) {
            out.copy_from_host(residual.data() + static_cast<std::size_t>(offset) * out_rows,
                               out.bytes);
        } else {
            cuda_check(cudaMemset(out.p, 0, out.bytes), "zero");
        }
        Tensor tx(static_cast<std::uint16_t*>(x.p) + static_cast<std::size_t>(offset) * k,
                  DType::BF16, {k, t});
        Tensor tout(out.p, DType::BF16, {out_rows, t});
        generic::Call call;
        call.x               = &tx;
        call.weight          = &device.weight;
        call.rows            = out_rows;
        call.up_offset       = gated(op) ? out_rows : 0;
        call.gelu_gate       = op == Op::GeGlu;
        call.relu2           = op == Op::Relu2;
        call.epilogue        = gated(op)       ? generic::Epilogue::SwiGlu
                               : op == Op::Add ? generic::Epilogue::AddResidual
                                               : generic::Epilogue::Store;
        call.out[0]          = &tout;
        call.segment_rows[0] = out_rows;
        generic::run_route(route, call, &workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "run");
        bits = download_bf16(out.p, static_cast<std::size_t>(out_rows) * t);
    };
    std::vector<std::uint16_t> reference;
    run(routes.front(), 0, wide, reference);
    int failures = 0;
    const std::pair<std::int32_t, std::int32_t> windows[] = {
        {0, 17}, {0, 31}, {0, 33}, {0, 64}, {0, 65}, {0, 129}, {5, 17}, {100, 200}, {283, 17}, {1, 299}};
    for (const auto route : routes) {
        for (const auto [offset, t] : windows) {
            std::vector<std::uint16_t> bits;
            run(route, offset, t, bits);
            for (std::int32_t token = 0; token < t; ++token) {
                if (std::memcmp(bits.data() + static_cast<std::size_t>(token) * out_rows,
                                reference.data() + static_cast<std::size_t>(offset + token) * out_rows,
                                static_cast<std::size_t>(out_rows) * 2) != 0) {
                    std::cerr << f.name << " " << op_name(op) << " invariance: route "
                              << generic::route_name(route) << " window " << offset << "+" << t
                              << " token " << token << " differs from the wide call\n";
                    ++failures;
                    break;
                }
            }
        }
    }
    if (failures == 0) {
        std::cout << f.name << " " << op_name(op) << (a4 ? " A4" : a8 ? " A8" : "") << " ["
                  << rows_n << "," << k
                  << "]: " << routes.size() << " routes x 10 windows byte-identical\n";
    }
    return failures;
}

// Volta's tensor-core verify route (volta_hmma_route, generic_volta_hmma.cu): the dispatch takes it
// for Q4_K / Q5_K / Q6_K / Q8 at 2..16 tokens (5..16 with an int8 route), and a token's output
// bytes equal those of the same token in a call of any other width there (at any start), repeated
// run to run: its sums read the token's own activation in one order.
int check_volta_hmma_invariance(const Format& f, Op op, std::int32_t out_rows, std::int32_t k,
                                ops::LinearPolicy policy, std::uint32_t seed) {
    const std::int32_t wide   = 40;
    const std::int32_t rows_n = gated(op) ? 2 * out_rows : out_rows;
    const HostWeight host     = make_weight(f.qtype, rows_n, k, seed);
    DeviceWeight device       = upload_weight(host);
    const auto activation     = make_linear_activation(k, wide, seed + 3U, ActivationCompute::A16);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    const auto residual = make_linear_activation(out_rows, wide, seed + 5U, ActivationCompute::A16);
    DeviceArena workspace(256);
    const generic::Problem problem{f.qtype, policy, out_rows, k,
                                   gated(op)       ? generic::Epilogue::SwiGlu
                                   : op == Op::Add ? generic::Epilogue::AddResidual
                                                   : generic::Epilogue::Store};
    const auto run = [&](std::int32_t offset, std::int32_t t, std::vector<std::uint16_t>& bits) {
        const generic::Route route = generic::select_route(problem, device.weight, t, false);
        if (route.family != generic::Family::RegisterMma) return false;
        DeviceBuffer out(static_cast<std::size_t>(out_rows) * t * 2);
        if (op == Op::Add) {
            out.copy_from_host(residual.data() + static_cast<std::size_t>(offset) * out_rows,
                               out.bytes);
        } else {
            cuda_check(cudaMemset(out.p, 0, out.bytes), "zero");
        }
        Tensor tx(static_cast<std::uint16_t*>(x.p) + static_cast<std::size_t>(offset) * k,
                  DType::BF16, {k, t});
        Tensor tout(out.p, DType::BF16, {out_rows, t});
        generic::Call call;
        call.x               = &tx;
        call.weight          = &device.weight;
        call.rows            = out_rows;
        call.up_offset       = gated(op) ? out_rows : 0;
        call.gelu_gate       = op == Op::GeGlu;
        call.relu2           = op == Op::Relu2;
        call.epilogue        = problem.epilogue;
        call.out[0]          = &tout;
        call.segment_rows[0] = out_rows;
        generic::run_route(route, call, &workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "run");
        bits = download_bf16(out.p, static_cast<std::size_t>(out_rows) * t);
        return true;
    };
    std::vector<std::uint16_t> reference, again;
    int failures = 0;
    if (!run(0, 16, reference) || !run(0, 16, again) || reference != again) {
        std::cerr << f.name << " " << op_name(op) << " Volta HMMA: T=16 not the register-MMA route"
                  << " or not repeatable\n";
        return 1;
    }
    const std::pair<std::int32_t, std::int32_t> windows[] = {
        {0, 2}, {3, 2}, {0, 3}, {5, 4}, {0, 7}, {9, 8}, {0, 9}, {1, 13}, {7, 15}, {0, 16}};
    for (const auto [offset, t] : windows) {
        // A call with an int8 route keeps it through four tokens (volta_hmma_route).
        if (policy == ops::LinearPolicy::AllowA8 && t <= 4) continue;
        std::vector<std::uint16_t> bits;
        if (!run(offset, t, bits)) {
            std::cerr << f.name << " " << op_name(op) << " Volta HMMA: T=" << t
                      << " not the register-MMA route\n";
            ++failures;
            continue;
        }
        for (std::int32_t token = 0; token < t; ++token) {
            if (offset + token >= 16) break;
            if (std::memcmp(bits.data() + static_cast<std::size_t>(token) * out_rows,
                            reference.data() + static_cast<std::size_t>(offset + token) * out_rows,
                            static_cast<std::size_t>(out_rows) * 2) != 0) {
                std::cerr << f.name << " " << op_name(op) << " Volta HMMA: window " << offset << "+"
                          << t << " token " << token << " differs from the T=16 call\n";
                ++failures;
                break;
            }
        }
    }
    if (failures == 0) {
        std::cout << f.name << " " << op_name(op) << " Volta HMMA [" << rows_n << "," << k
                  << "]: 10 windows byte-identical\n";
    }
    return failures;
}

// An R8 route: every token's output bytes equal those of the same token in a call of any other
// T <= kR8MaxTokens (at any start), since a token's sums read only its own activation; with the
// workspace the verify widths of r8_verify_compiled formats read the activation quantized once
// per call, which holds the same bytes as the CTA's own quantization.
int check_r8_invariance(const Format& f, Op op, std::int32_t out_rows, std::int32_t k,
                        generic::Route route, std::uint32_t seed) {
    const std::int32_t wide   = generic::kR8MaxTokens;
    const std::int32_t rows_n = gated(op) ? 2 * out_rows : out_rows;
    const HostWeight host     = make_weight(f.qtype, rows_n, k, seed);
    DeviceWeight device       = upload_weight(host);
    const auto activation     = make_linear_activation(k, wide, seed + 3U, ActivationCompute::A16);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    DeviceArena workspace(
        generic::route_workspace_bytes(generic::Route{generic::Family::W4a8Mma, 0}, k, wide));
    const auto run = [&](std::int32_t offset, std::int32_t t, std::vector<std::uint16_t>& bits) {
        DeviceBuffer out(static_cast<std::size_t>(out_rows) * t * 2);
        cuda_check(cudaMemset(out.p, 0, out.bytes), "zero");
        Tensor tx(static_cast<std::uint16_t*>(x.p) + static_cast<std::size_t>(offset) * k,
                  DType::BF16, {k, t});
        Tensor tout(out.p, DType::BF16, {out_rows, t});
        generic::Call call;
        call.x               = &tx;
        call.weight          = &device.weight;
        call.rows            = out_rows;
        call.up_offset       = gated(op) ? out_rows : 0;
        call.gelu_gate       = op == Op::GeGlu;
        call.relu2           = op == Op::Relu2;
        call.epilogue        = gated(op) ? generic::Epilogue::SwiGlu : generic::Epilogue::Store;
        call.out[0]          = &tout;
        call.segment_rows[0] = out_rows;
        generic::run_route(route, call, &workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "run");
        bits = download_bf16(out.p, static_cast<std::size_t>(out_rows) * t);
    };
    std::vector<std::uint16_t> reference;
    run(0, wide, reference);
    int failures = 0;
    const std::pair<std::int32_t, std::int32_t> windows[] = {{0, 1}, {3, 1}, {1, 1}, {0, 2},
                                                             {1, 3}, {2, 2}, {0, 3}, {1, 2}};
    for (const auto [offset, t] : windows) {
        std::vector<std::uint16_t> bits;
        run(offset, t, bits);
        for (std::int32_t token = 0; token < t; ++token) {
            if (std::memcmp(bits.data() + static_cast<std::size_t>(token) * out_rows,
                            reference.data() + static_cast<std::size_t>(offset + token) * out_rows,
                            static_cast<std::size_t>(out_rows) * 2) != 0) {
                std::cerr << f.name << " " << op_name(op) << " R8 invariance: route "
                          << generic::route_name(route) << " window " << offset << "+" << t
                          << " token " << token << " differs from T=" << wide << "\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

// Every R8 schedule gives the same bytes: a lane sums the block groups lane, lane + 32, ... of its
// row in one order whatever the warps per CTA and rows per warp, so heuristic_r8 may pick any of
// them. K spans more than 64 groups (both ring slots, several loop trips); N is no multiple of
// any schedule's rows per CTA; T = 2..4 takes the pre-quantized form where the format has one.
int check_r8_schedules_agree(const Format& f, Op op, std::int32_t out_rows, std::int32_t k,
                             std::uint32_t seed) {
    const std::int32_t wide   = generic::kR8MaxTokens;
    const std::int32_t rows_n = gated(op) ? 2 * out_rows : out_rows;
    const HostWeight host     = make_weight(f.qtype, rows_n, k, seed);
    DeviceWeight device       = upload_weight(host);
    const auto activation     = make_linear_activation(k, wide, seed + 3U, ActivationCompute::A16);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    DeviceArena workspace(
        generic::route_workspace_bytes(generic::Route{generic::Family::W4a8Mma, 0}, k, wide));
    // The add epilogue reads a residual: the same one for every schedule.
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(out_rows) * wide);
    {
        std::mt19937 rng(seed + 11U);
        std::uniform_real_distribution<float> u(-0.5F, 0.5F);
        for (auto& v : residual) v = f32_to_bf16(u(rng));
    }
    const auto run = [&](generic::Route route, std::int32_t t) {
        DeviceBuffer out(static_cast<std::size_t>(out_rows) * t * 2);
        out.copy_from_host(residual.data(), out.bytes);
        Tensor tx(x.p, DType::BF16, {k, t});
        Tensor tout(out.p, DType::BF16, {out_rows, t});
        generic::Call call;
        call.x               = &tx;
        call.weight          = &device.weight;
        call.rows            = out_rows;
        call.up_offset       = gated(op) ? out_rows : 0;
        call.gelu_gate       = op == Op::GeGlu;
        call.relu2           = op == Op::Relu2;
        call.epilogue        = gated(op)       ? generic::Epilogue::SwiGlu
                               : op == Op::Add ? generic::Epilogue::AddResidual
                                               : generic::Epilogue::Store;
        call.out[0]          = &tout;
        call.segment_rows[0] = out_rows;
        generic::run_route(route, call, &workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "run");
        return download_bf16(out.p, static_cast<std::size_t>(out_rows) * t);
    };
    int failures = 0;
    for (std::int32_t t = 1; t <= wide; ++t) {
        const bool pre =
            t >= 2 && t <= generic::kR8VerifyTokens && generic::r8_verify_compiled(f.qtype);
        if (!pre && !generic::r8_fits(f.qtype, k, t)) continue;
        const auto reference = run(generic::Route{generic::Family::R8Mma, 0}, t);
        for (int i = 1; i < generic::r8_schedule_count(); ++i) {
            const generic::Route r{generic::Family::R8Mma, static_cast<std::uint8_t>(i)};
            if (run(r, t) != reference) {
                std::cerr << f.name << " " << op_name(op) << " [" << rows_n << "," << k
                          << "] T=" << t << ": R8 route " << generic::route_name(r)
                          << " differs from " << generic::route_name({generic::Family::R8Mma, 0})
                          << "\n";
                ++failures;
            }
        }
    }
    if (failures == 0) {
        std::cout << f.name << " " << op_name(op) << " [" << rows_n << "," << k << "]: "
                  << generic::r8_schedule_count() << " R8 schedules byte-identical\n";
    }
    return failures;
}

// The invariant route profile (--activations exact): every call through select_route, a single
// token included and whatever the policy allows, gives a token the bytes of the wide call.
int check_invariant_profile(const Format& f, Op op, std::int32_t out_rows, std::int32_t k,
                            std::uint32_t seed) {
    const std::int32_t wide   = 300;
    const std::int32_t rows_n = gated(op) ? 2 * out_rows : out_rows;
    const HostWeight host     = make_weight(f.qtype, rows_n, k, seed);
    DeviceWeight device       = upload_weight(host);
    const auto activation     = make_linear_activation(k, wide, seed + 3U, ActivationCompute::A16);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    DeviceArena workspace(std::max<std::size_t>(
        std::max(generic::route_workspace_bytes(generic::Route{generic::Family::W4a4Mma, 0}, k, wide),
                 generic::route_workspace_bytes(generic::Route{generic::Family::W4a8Mma, 0}, k, wide)),
        256));
    const ops::LinearPolicy policy = f.qtype == QType::NVFP4 ? ops::LinearPolicy::AllowA4
                                     : generic::w4a8_supports(f.qtype) ? ops::LinearPolicy::AllowA8
                                                                       : ops::LinearPolicy::A16Only;
    const auto run = [&](std::int32_t offset, std::int32_t t, std::vector<std::uint16_t>& bits) {
        DeviceBuffer out(static_cast<std::size_t>(out_rows) * t * 2);
        cuda_check(cudaMemset(out.p, 0, out.bytes), "zero");
        Tensor tx(static_cast<std::uint16_t*>(x.p) + static_cast<std::size_t>(offset) * k,
                  DType::BF16, {k, t});
        Tensor tout(out.p, DType::BF16, {out_rows, t});
        generic::Call call;
        call.x               = &tx;
        call.weight          = &device.weight;
        call.rows            = out_rows;
        call.up_offset       = gated(op) ? out_rows : 0;
        call.gelu_gate       = op == Op::GeGlu;
        call.relu2           = op == Op::Relu2;
        call.epilogue        = gated(op) ? generic::Epilogue::SwiGlu : generic::Epilogue::Store;
        call.out[0]          = &tout;
        call.segment_rows[0] = out_rows;
        generic::run(call, policy, &workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "run");
        bits = download_bf16(out.p, static_cast<std::size_t>(out_rows) * t);
    };
    int failures = 0;
    for (const bool prefill : {false, true}) {
        const ops::ScopedPrefillRoutes profile(prefill);
        const ops::ScopedInvariantRoutes invariant(true);
        std::vector<std::uint16_t> reference;
        run(0, wide, reference);
        const std::pair<std::int32_t, std::int32_t> windows[] = {
            {0, 1}, {7, 1}, {299, 1}, {0, 2}, {3, 4}, {0, 8}, {1, 9},
            {5, 16}, {0, 17}, {40, 33}, {0, 64}, {100, 200}, {283, 17}};
        for (const auto [offset, t] : windows) {
            std::vector<std::uint16_t> bits;
            run(offset, t, bits);
            if (std::memcmp(bits.data(),
                            reference.data() + static_cast<std::size_t>(offset) * out_rows,
                            bits.size() * 2) != 0) {
                std::cerr << f.name << " " << op_name(op) << " K=" << k
                          << " invariant profile" << (prefill ? " (prefill)" : "") << ": window "
                          << offset << "+" << t << " differs from the wide call\n";
                ++failures;
            }
        }
    }
    if (failures == 0) {
        std::cout << f.name << " " << op_name(op) << " [" << rows_n << "," << k
                  << "]: invariant profile, 13 windows x 2 byte-identical\n";
    }
    return failures;
}

// The NVFP4 tiled route of each class count against the register MMA with that K split: the same
// bytes for every epilogue, on a row range that starts inside a 128-row scale tile and is not a
// whole number of tiles, at widths from one token to several token tiles.
int check_nvfp4_tiled_order(const Format& f, Op op, std::int32_t k, std::uint32_t seed) {
    const std::int32_t weight_n  = 512;
    const std::int32_t row_begin = gated(op) ? 40 : 64;
    const std::int32_t rows      = 200;
    const std::int32_t up_offset = gated(op) ? 256 : 0;
    const std::int32_t wide      = 200;
    const HostWeight host        = make_weight(f.qtype, weight_n, k, seed);
    DeviceWeight device          = upload_weight(host);
    const auto activation = make_linear_activation(k, wide, seed + 5U, ActivationCompute::A16);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(rows) * wide);
    {
        std::mt19937 rng(seed + 13U);
        std::uniform_real_distribution<float> u(-0.5F, 0.5F);
        for (auto& v : residual) v = f32_to_bf16(u(rng));
    }
    const auto rmma_route = [](int warps_m, int warps_k, int m_tiles, int n_tiles) {
        for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
            const auto s = generic::rmma_schedule_info(i);
            if (s.warps_m == warps_m && s.warps_k == warps_k && s.m_tiles == m_tiles &&
                s.n_tiles == n_tiles) {
                return generic::Route{generic::Family::RegisterMma, static_cast<std::uint8_t>(i)};
            }
        }
        throw std::logic_error("no such register MMA schedule");
    };
    const auto run = [&](generic::Route route, std::int32_t t, std::vector<std::uint16_t>& bits) {
        DeviceBuffer out(static_cast<std::size_t>(rows) * t * 2);
        out.copy_from_host(residual.data(), out.bytes);
        Tensor tx(x.p, DType::BF16, {k, t});
        Tensor tout(out.p, DType::BF16, {rows, t});
        generic::Call call;
        call.x               = &tx;
        call.weight          = &device.weight;
        call.row_begin       = row_begin;
        call.rows            = rows;
        call.up_offset       = up_offset;
        call.gelu_gate       = op == Op::GeGlu;
        call.relu2           = op == Op::Relu2;
        call.epilogue        = (op == Op::Linear || op == Op::Relu2) ? generic::Epilogue::Store
                               : op == Op::Add ? generic::Epilogue::AddResidual
                                               : generic::Epilogue::SwiGlu;
        call.out[0]          = &tout;
        call.segment_rows[0] = rows;
        generic::run_route(route, call, nullptr, nullptr);
        cuda_check(cudaDeviceSynchronize(), "run");
        bits = download_bf16(out.p, static_cast<std::size_t>(rows) * t);
    };
    int failures = 0;
    for (int i = 0; kNativeNvfp4 && i < generic::nvfp4_tiled_schedule_count(); ++i) {
        const generic::Route tiled{generic::Family::Nvfp4TiledMma, static_cast<std::uint8_t>(i)};
        const int classes = generic::nvfp4_tiled_schedule_info(i).classes;
        const generic::Route rmma =
            classes == 1 ? rmma_route(4, 1, 1, 8) : rmma_route(1, classes, 1, 4);
        for (const std::int32_t t : {1, 17, 64, 65, 200}) {
            std::vector<std::uint16_t> want, got;
            run(rmma, t, want);
            run(tiled, t, got);
            if (want != got) {
                std::cerr << f.name << " " << op_name(op) << " K=" << k << " route "
                          << generic::route_name(tiled) << " T=" << t << " differs from "
                          << generic::route_name(rmma) << '\n';
                ++failures;
            }
        }
    }
    if (failures == 0) {
        std::cout << f.name << " " << op_name(op) << " K=" << k << ": "
                  << generic::nvfp4_tiled_schedule_count()
                  << " tiled routes x 5 widths byte-identical to the register MMA\n";
    }
    return failures;
}

} // namespace

// IQ3_XXS with its sign plane 4 bytes off an 8-byte boundary: route admission refuses the
// producer-warp tile (it reads the plane 8 bytes at a time), the picker takes a route that admits it
// tile (on a grid that fills the GPU), and a producer-warp route named anyway runs that classic
// tile; every one of them writes the bytes of the aligned weight on the producer-warp tile. A
// build without the producer-warp ring (sm_75) compares with the classic tile on the aligned plane
// and forces no producer route.
int check_w4a8_misaligned_signs(const Format& f) {
    const std::int32_t n = 136, k = 1280, t = 200;
    const HostWeight host = make_weight(f.qtype, n, k, 81U);
    DeviceWeight device   = upload_weight(host);
    const std::size_t high_bytes =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(k / 256) * 32U;
    DeviceBuffer shifted(high_bytes + 16);
    auto* high = static_cast<std::uint8_t*>(shifted.p) + 4; // cudaMalloc: 256-byte aligned
    cuda_check(cudaMemcpy(high, device.weight.qhigh, high_bytes, cudaMemcpyDeviceToDevice),
               "shift the sign plane");
    Weight moved = device.weight;
    moved.qhigh  = high;

    generic::Route ws{generic::Family::W4a8Mma, 0}, classic{generic::Family::W4a8Mma, 0};
    for (int i = 0; i < generic::w4a8_schedule_count(); ++i) {
        const auto s = generic::w4a8_schedule_info(i);
        if (s.block_m == 128 && s.block_n == 64) {
            (s.producers > 0 ? ws : classic).schedule = static_cast<std::uint8_t>(i);
        }
    }
    const auto policy = ops::LinearPolicy::AllowA8;
    int failures      = 0;
    const auto expect = [&](bool ok, const char* what) {
        if (!ok) {
            std::cerr << f.name << " misaligned sign plane: " << what << "\n";
            ++failures;
        }
    };
    // The producer-warp tile exists only where its ring fits (not in the sm_75 build).
    if (generic::w4a8_fits(ws.schedule, f.qtype)) {
        expect(generic::route_admits(ws, device.weight, policy), "aligned weight refused");
    }
    expect(!generic::route_admits(ws, moved, policy), "misaligned weight admitted");
    expect(generic::route_admits(classic, moved, policy), "classic 128x64 refused");
    // The pick at a row count whose 128-token grid gives every SM a CTA (at 136 rows the grid's
    // 6 CTAs take 64x64 on any GPU); the weight's sign plane is what it judges. The problem's
    // rows need not match the uploaded weight's: admission reads the weight's planes only.
    const std::int32_t pick_rows = static_cast<std::int32_t>(64 * device_class().sms);
    const generic::Problem problem{f.qtype, policy, pick_rows, k, generic::Epilogue::Store};
    const generic::Route picked = generic::heuristic_route(problem, moved, t, false);
    expect(generic::route_admits(picked, moved, policy), "the picker takes a route that refuses it");
    std::cout << f.name << " misaligned sign plane: picked " << generic::route_name(picked) << std::endl;
    expect(!(picked == ws) || !generic::w4a8_fits(ws.schedule, f.qtype),
           "the picker takes the producer-warp tile");

    const auto activation = make_linear_activation(k, t, 83U, ActivationCompute::A8);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    DeviceArena workspace(std::max<std::size_t>(
        generic::route_workspace_bytes(generic::Route{generic::Family::W4a8Mma, 0}, k, t), 256));
    const auto run = [&](generic::Route route, const Weight& weight) {
        DeviceBuffer out(static_cast<std::size_t>(n) * t * 2);
        cuda_check(cudaMemset(out.p, 0, out.bytes), "zero");
        Tensor tx(x.p, DType::BF16, {k, t});
        Tensor tout(out.p, DType::BF16, {n, t});
        generic::Call call;
        call.x               = &tx;
        call.weight          = &weight;
        call.rows            = n;
        call.epilogue        = generic::Epilogue::Store;
        call.out[0]          = &tout;
        call.segment_rows[0] = n;
        generic::run_route(route, call, &workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "run");
        return download_bf16(out.p, static_cast<std::size_t>(n) * t);
    };
    const bool ring      = generic::w4a8_fits(ws.schedule, f.qtype);
    const auto aligned   = ring ? ws : classic;
    const auto reference = run(aligned, device.weight);
    // A W4A8 pick writes the bytes of every W4A8 tile; another family its own on the aligned plane.
    const bool picked_w4a8 = picked.family == generic::Family::W4a8Mma;
    expect(run(picked, moved) == (picked_w4a8 ? reference : run(picked, device.weight)),
           "the picked route writes other bytes");
    if (ring) {
        expect(run(ws, moved) == reference, "a forced producer-warp route writes other bytes");
    }
    if (failures == 0) {
        std::cout << f.name << " misaligned sign plane: " << generic::route_name(picked)
                  << ", byte-identical to " << generic::route_name(aligned)
                  << " on the aligned plane\n";
    }
    return failures;
}

// A q8_1 block of large activations (SwiGLU outputs reach thousands) sums past fp16's range: its
// (scale, sum) halves must stay finite, or every format whose dot reads the sum (a minimum or an
// offset: Q4_1, Q5_1, Q2_K, Q4_K, Q5_K) turns the token's whole output into NaN.
int check_mmq_outlier_block(const Format& f) {
    const std::int32_t n = 128;
    const std::int32_t k = 512;
    const std::int32_t t = 8;
    const generic::Route mmq{generic::Family::GgmlMmq, 0};
    Weight weight = cached_weight(f.qtype, n, k, 5U).device->weight;
    if (!generic::route_admits(mmq, weight, ops::LinearPolicy::AllowA8)) {
        std::cout << f.name << " ggml_mmq outlier block: not admitted on this device\n";
        return 0;
    }
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * t);
    std::mt19937 rng(41U);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    for (auto& v : activation) { v = f32_to_bf16(u(rng)); }
    // Token 3: one block of 32 values at 4000, a sum of 128000 (fp16 ends at 65504).
    for (std::int32_t j = 64; j < 96; ++j) {
        activation[static_cast<std::size_t>(3) * k + j] = f32_to_bf16(4000.0F);
    }
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    DeviceBuffer out(static_cast<std::size_t>(n) * t * 2);
    Tensor tx(x.p, DType::BF16, {k, t});
    Tensor tout(out.p, DType::BF16, {n, t});
    DeviceArena workspace(std::max<std::size_t>({generic::route_workspace_bytes(mmq, k, t),
                                                 generic::ggml_mmq_workspace_bytes(k, n, t), 256}));
    generic::Call call;
    call.x                        = &tx;
    call.weight                   = &weight;
    call.rows                     = n;
    call.epilogue                 = generic::Epilogue::Store;
    call.out[0]                   = &tout;
    call.segment_rows[0]          = n;
    const std::uint64_t fallbacks = generic::ggml_mmq_fallback_count();
    try {
        generic::run_route(mmq, call, &workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "outlier run");
    } catch (const std::exception& e) {
        std::cerr << f.name << " ggml_mmq outlier block: unexpected exception: " << e.what()
                  << '\n';
        return 1;
    }
    if (generic::ggml_mmq_fallback_count() != fallbacks) {
        std::cerr << f.name << " ggml_mmq outlier block: ran A16, not ggml_mmq\n";
        return 1;
    }
    int non_finite = 0;
    float largest  = 0.0F;
    for (const std::uint16_t bits : download_bf16(out.p, static_cast<std::size_t>(n) * t)) {
        if ((bits & 0x7f80U) == 0x7f80U) {
            ++non_finite;
        } else {
            largest = std::max(largest, std::fabs(bf16_to_f32(bits)));
        }
    }
    std::cout << f.name << " ggml_mmq outlier block: max |y| " << largest << ", non-finite "
              << non_finite << "\n";
    if (non_finite != 0) {
        std::cerr << f.name << " ggml_mmq: a q8_1 block summing past fp16 made " << non_finite
                  << " non-finite outputs\n";
        return 1;
    }
    return 0;
}

// ggml's matrix kernel on Q4_K / Q5_K where the offset term carries the product: centred weights
// (m = sc and dmin = 7.5 d, so w = d sc (q - 7.5)) against activation blocks of one outlier (60)
// among values of 2 +- 1, as a layer's input with a few large channels. The codes' dot and the
// offset term are then both large and only their difference is the product, so both must see the
// int8 activation: the oracle is the FP64 product of the independently decoded weights and the
// activation quantized per 32 values (quantize_a8). With the block sum of the raw values (ggml's own
// quantizer) the quantization noise of the dot stayed in the product.
int check_mmq_code_sums(const Format& f) {
    const std::int32_t n = 256, k = 2048, t = 64;
    const generic::Route mmq{generic::Family::GgmlMmq, 0};
    HostWeight host               = make_weight(f.qtype, n, k, 61U);
    const std::size_t block_bytes = f.qtype == QType::GGML_Q4_K ? 144 : 176;
    auto& bytes                   = host.packed.payload;
    for (std::size_t b = 0; b + block_bytes <= bytes.size(); b += block_bytes) {
        // The 12 scale bytes after d and dmin: m_j = sc_j for every sub-block.
        std::uint8_t* sc = bytes.data() + b + 4;
        for (int j = 0; j < 4; ++j) { sc[4 + j] = sc[j]; }
        for (int j = 8; j < 12; ++j) { sc[j] = static_cast<std::uint8_t>((sc[j] & 0x0FU) * 0x11U); }
        const float d = quantized_weight::detail::f16_to_f32(quantized_weight::detail::load_u16_le(bytes, b));
        quantized_weight::detail::store_u16_le(bytes, b + 2,
                                               quantized_weight::detail::f32_to_f16(
                                                   (f.qtype == QType::GGML_Q4_K ? 7.5F : 15.5F) * d));
    }
    DeviceWeight device = upload_weight(host);
    const Weight& weight = device.weight;
    const auto policy    = ops::LinearPolicy::AllowA8;
    if (!generic::route_admits(mmq, weight, policy)) {
        std::cout << f.name << " ggml_mmq code sums: not admitted on this device\n";
        return 0;
    }
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * t);
    std::mt19937 rng(67U);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    for (std::size_t i = 0; i < activation.size(); ++i) {
        activation[i] = f32_to_bf16(i % 32 == 5 ? 60.0F : 2.0F + u(rng));
    }
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    DeviceArena workspace(std::max<std::size_t>({generic::route_workspace_bytes(mmq, k, t),
                                                 generic::ggml_mmq_workspace_bytes(k, n, t), 256}));
    const auto run = [&](generic::Route route) {
        DeviceBuffer out(static_cast<std::size_t>(n) * t * 2);
        Tensor tx(x.p, DType::BF16, {k, t});
        Tensor tout(out.p, DType::BF16, {n, t});
        generic::Call call;
        call.x               = &tx;
        call.weight          = &weight;
        call.rows            = n;
        call.epilogue        = generic::Epilogue::Store;
        call.out[0]          = &tout;
        call.segment_rows[0] = n;
        workspace.reset();
        generic::run_route(route, call, &workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "code-sum run");
        return download_bf16(out.p, static_cast<std::size_t>(n) * t);
    };
    const std::uint64_t fallbacks = generic::ggml_mmq_fallback_count();
    const auto got                = run(mmq);
    if (generic::ggml_mmq_fallback_count() != fallbacks) {
        std::cerr << f.name << " ggml_mmq code sums: ran A16, not ggml_mmq\n";
        return 1;
    }
    std::vector<std::int32_t> rows(static_cast<std::size_t>(n));
    for (std::int32_t i = 0; i < n; ++i) { rows[static_cast<std::size_t>(i)] = i; }
    const std::vector<float> w = weight_rows(host, rows);
    std::vector<float> xq(activation.size());
    for (std::size_t i = 0; i < xq.size(); ++i) { xq[i] = bf16_to_f32(activation[i]); }
    quantize_a8(xq, k);
    std::vector<double> oracle(static_cast<std::size_t>(n) * t);
    cpu_linear_gemm_fp64(w.data(), xq.data(), oracle.data(), n, k, t);
    double err = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double a = bf16_to_f32(got[i]);
        err += (a - oracle[i]) * (a - oracle[i]);
        norm += oracle[i] * oracle[i];
    }
    const double rel = std::sqrt(err / norm);
    std::cout << f.name << " ggml_mmq code sums: rel_l2 to the int8 activation's FP64 oracle " << rel
              << "\n";
    if (!(rel < kMmqCodeSumAgreement)) {
        std::cerr << f.name << " ggml_mmq on centred weights: rel_l2 " << rel << " to the oracle (at most "
                  << kMmqCodeSumAgreement << ")\n";
        return 1;
    }
    return 0;
}

// NVFP4 under a policy that allows A8 but not A4 (a W4A16 checkpoint under --activations int8):
// the generic W4A8 from 17 tokens on every build, the tiled route there without tensor mma.sync
// (Volta: volta_tiled_route), decode and verify widths unchanged; the same
// weight under AllowA4 (it carries its activation divisor) keeps W4A4 where the build has it and
// is never W4A8's there, nor is it without a divisor (A16 then, as before). Through the public
// entry points the shapes the native NVFP4 plans own (sm_120a: linear 1024x2816, linear_add
// 5120x6144, linear_swiglu 34816x5120) leave their plans for it inside a prefill only: the call
// quantizes x into the W4A8 planes of the workspace the capacity query sized, and answers against
// the oracle of the int8 x; outside one (a verify of 17 columns or more) the plan runs and reads no
// planes.
int check_nvfp4_a8_routes() {
    int failures          = 0;
    const Format& nvfp4   = format_of(QType::NVFP4);
    const auto a8         = ops::LinearPolicy::AllowA8;
    const auto a4         = ops::LinearPolicy::AllowA4;
    const HostWeight host = make_weight(QType::NVFP4, 384, 1024, 43U);
    const DeviceWeight dw = upload_weight(host);
    Weight no_divisor              = dw.weight;
    no_divisor.input_scale_divisor = 0.0F;
    const generic::Problem p8{QType::NVFP4, a8, 384, 1024, generic::Epilogue::Store};
    const generic::Problem p4{QType::NVFP4, a4, 384, 1024, generic::Epilogue::Store};
    for (const std::int32_t t : {1, 4, 16, 17, 64, 200}) {
        const generic::Route r8 = generic::select_route(p8, dw.weight, t, false);
        const generic::Route r4 = generic::select_route(p4, dw.weight, t, false);
        const generic::Route r0 = generic::select_route(p4, no_divisor, t, false);
        const bool tiled        = generic::volta_tiled_route(dw.weight, a8, t).has_value();
        const bool wide         = t >= 17 && !tiled;
        const ops::ScopedPrefillRoutes prefill(true);
        if ((r8.family == generic::Family::W4a8Mma) != wide ||
            (tiled && r8.family != generic::Family::TiledMma) ||
            generic::owned_prefill_w4a8(dw.weight, a8, t) != wide) {
            std::cerr << "nvfp4 A8 T=" << t << ": " << generic::route_name(r8) << "\n";
            ++failures;
        }
        if (kNativeNvfp4 && (r4.family == generic::Family::W4a8Mma ||
                             generic::owned_prefill_w4a8(dw.weight, a4, t) ||
                             (wide && r4.family != generic::Family::W4a4Mma))) {
            std::cerr << "nvfp4 A4 T=" << t << ": " << generic::route_name(r4) << "\n";
            ++failures;
        }
        if (kNativeNvfp4 &&
            (r0.family == generic::Family::W4a8Mma || r0.family == generic::Family::W4a4Mma ||
             generic::owned_prefill_w4a8(no_divisor, a4, t))) {
            std::cerr << "nvfp4 A4 without divisor T=" << t << ": " << generic::route_name(r0)
                      << "\n";
            ++failures;
        }
    }
    {
        const ops::ScopedPrefillRoutes speed(false);
        if (generic::owned_prefill_w4a8(dw.weight, a8, 64)) {
            std::cerr << "nvfp4 A8: owned_prefill_w4a8 outside a prefill\n";
            ++failures;
        }
    }
    for (int i = 0; kNativeNvfp4 && i < generic::w4a8_schedule_count(); ++i) {
        const generic::Route r{generic::Family::W4a8Mma, static_cast<std::uint8_t>(i)};
        if (generic::route_admits(r, dw.weight, a4) || generic::route_admits(r, no_divisor, a4)) {
            std::cerr << "nvfp4 A4: admitted to " << generic::route_name(r) << "\n";
            ++failures;
        }
    }
    if (!kNativeNvfp4) return failures;

    struct Owned {
        Op op;
        std::int32_t n, k;
    };

    for (const Owned o : {Owned{Op::Linear, 1024, 2816}, Owned{Op::Add, 5120, 6144},
                          Owned{Op::SwiGlu, 17408, 5120}}) {
        const std::int32_t rows = gated(o.op) ? 2 * o.n : o.n;
        for (const auto& [t, in_prefill] :
             {std::pair{16, true}, std::pair{17, true}, std::pair{64, true}, std::pair{17, false},
              std::pair{33, false}}) {
            const ops::ScopedPrefillRoutes profile(in_prefill);
            const bool expect_w4a8 = t >= 17 && in_prefill;
            if (expect_w4a8) {
                failures +=
                    check_case(nvfp4, o.op, o.n, o.k, t, a8, 47U, nullptr, ActivationCompute::A8);
            }
            const Weight& w       = cached_weight(QType::NVFP4, rows, o.k, 47U).device->weight;
            const auto activation = make_linear_activation(o.k, t, 53U, ActivationCompute::A16);
            DeviceBuffer x(activation.size() * 2);
            x.copy_from_host(activation.data(), x.bytes);
            DeviceBuffer out(static_cast<std::size_t>(o.n) * t * 2);
            out.fill(0); // settles before the launch
            Tensor tx(x.p, DType::BF16, {o.k, t});
            Tensor tout(out.p, DType::BF16, {o.n, t});
            const std::size_t capacity =
                o.op == Op::Linear
                    ? ops::linear_workspace_capacity_bytes(QType::NVFP4, o.n, o.k, a8, t, t)
                : o.op == Op::Add
                    ? ops::linear_add_workspace_capacity_bytes(QType::NVFP4, o.n, o.k, a8, t, t)
                    : ops::linear_swiglu_workspace_capacity_bytes(QType::NVFP4, rows, o.k, a8, t,
                                                                  t);
            DeviceArena workspace(std::max<std::size_t>(capacity, 256));
            if (o.op == Op::Linear) {
                ops::linear(tx, w, tout, a8, workspace, nullptr);
            } else if (o.op == Op::Add) {
                ops::linear_add(tx, w, tout, a8, workspace, nullptr);
            } else {
                ops::linear_swiglu(tx, w, tout, a8, workspace, nullptr);
            }
            cuda_check(cudaDeviceSynchronize(), "nvfp4 A8 owned shape");
            const std::size_t planes =
                generic::route_workspace_bytes(generic::Route{generic::Family::W4a8Mma, 0}, o.k, t);
            // The W4A8 call takes exactly its planes; the plan none (linear, linear_add) or its
            // projected gate/up plane (linear_swiglu past 16 columns).
            const bool took_w4a8 = workspace.peak_used() == planes;
            if (took_w4a8 != expect_w4a8) {
                std::cerr << "nvfp4 A8 " << op_name(o.op) << " [" << rows << "," << o.k
                          << "] T=" << t << (in_prefill ? " prefill" : " speed")
                          << ": workspace use " << workspace.peak_used()
                          << " against the W4A8 planes' " << planes << "\n";
                ++failures;
            }
        }
    }
    if (failures == 0) {
        std::cout << "nvfp4 A8 (no A4): W4A8 from 17 tokens in a prefill, the native plans "
                     "elsewhere; A4 keeps W4A4 (A16 without a divisor)\n";
    }
    return failures;
}

int main() {
    if (!cuda_available()) {
        std::cout << "CUDA unavailable; skipping generic linear test\n";
        return 77;
    }
    if (const char* v = std::getenv("NINFERNO_TEST_FORMATS"); v != nullptr) {
        int matched = 0;
        for (const Format& f : kFormats) matched += on_build(f) ? 1 : 0;
        std::cout << "generic linear: NINFERNO_TEST_FORMATS=" << v << " (" << matched
                  << " format(s) on this build)\n";
        if (matched == 0) {
            std::cerr << "generic linear: NINFERNO_TEST_FORMATS matches no format\n";
            return 1;
        }
    }
    int failures = 0;
    std::mt19937 rng(20260924U);

    // 0. The wide-load Q6_K's choice (rmma_wide_rule) on the boards' facts: DRAM bytes per SM
    // cycle and compute capability; as many CTAs and as much local memory as the plain kernel.
    {
        namespace g = ops::detail::generic;

        struct Board {
            const char* name;
            double bytes;
            int cc;
            bool wide;
        };

        const Board boards[] = {{"RTX 4090", 3.1, 89, true},     {"RTX 4070 Ti", 3.2, 89, true},
                                {"RTX 5090", 4.4, 120, true},    {"RTX 5070 Ti", 5.2, 120, true},
                                {"RTX 2080 Ti", 5.9, 75, false}, {"RTX 3060", 7.2, 86, false},
                                {"Tesla V100", 3.4, 70, false},  {"facts unknown", 0.0, 89, false}};
        for (const Board& b : boards) {
            if (g::rmma_wide_rule(b.bytes, b.cc, true, 0, 0) != b.wide) {
                std::cerr << "rmma_wide_rule: " << b.name << " expected " << b.wide << "\n";
                ++failures;
            }
        }
        // Occupancy lost (rmma_variant_occupancy_rule), or a spill the plain kernel lacks: plain,
        // on any board.
        if (g::rmma_wide_rule(3.1, 89, false, 0, 0) || g::rmma_wide_rule(3.1, 89, true, 64, 0)) {
            std::cerr << "rmma_wide_rule: took the wide kernel at lower occupancy or a spill\n";
            ++failures;
        }
        if (!g::rmma_wide_rule(3.1, 89, true, 16, 16) || !g::rmma_wide_rule(3.1, 89, true, 8, 0) ||
            g::rmma_wide_rule(3.1, 89, true, 160, 0)) {
            std::cerr << "rmma_wide_rule: refused a spill within the slack or took a large one\n";
            ++failures;
        }
    }

    // 0b. A variant kernel's occupancy (rmma_variant_occupancy_rule): fewer resident CTAs per SM
    // than the plain kernel's cost only where the call's grid takes another wave at them.
    {
        namespace g = ops::detail::generic;

        struct Case {
            std::int64_t grid;
            int sms;
            int variant;
            int plain;
            bool kept;
        };

        const Case cases[] = {
            {256, 70, 4, 4, true},    // as many CTAs: always
            {256, 70, 3, 4, false},   // 70 SMs, 256 CTAs: one wave at 4 (280), two at 3 (210)
            {192, 70, 3, 4, true},    // 192 CTAs: one wave at either
            {600, 70, 3, 4, true},    // three waves at either (280 and 210 a wave)
            {768, 70, 3, 4, false},   // three waves at 4, four at 3
            {1088, 128, 2, 3, false}, // 128 SMs, 1088 CTAs: three waves at 3, five at 2
            {80, 68, 1, 2, true},     // fewer CTAs than SMs: one wave at either
            {80, 68, 0, 2, false},    // the variant fits no CTA
            {80, 68, 2, 0, false},    // counts unknown
            {80, 0, 2, 2, false},     // SMs unknown
        };
        for (const Case& c : cases) {
            if (g::rmma_variant_occupancy_rule(c.grid, c.sms, c.variant, c.plain) != c.kept) {
                std::cerr << "rmma_variant_occupancy_rule: grid " << c.grid << " on " << c.sms
                          << " SMs at " << c.variant << " against " << c.plain << " expected "
                          << c.kept << "\n";
                ++failures;
            }
        }
    }

    // 1. Public entry points at random and curated awkward shapes, every format and op.
    for (const Format& f : kFormats) {
        if (!on_build(f)) continue;
        std::vector<std::pair<std::int32_t, std::int32_t>> shapes;
        if (f.qtype == QType::NVFP4) {
            // Any N (200, 1000, 72: a last 128-row scale tile the rows do not fill, as a vocabulary
            // of 202048 rows ends); K whole 64-value tiles.
            shapes = {{128, 64}, {384, 1088}, {1152, 960}, {256, 3840}, {200, 1088}, {1000, 960},
                      {72, 256}};
            for (int i = 0; i < 3; ++i) {
                shapes.push_back({128 * static_cast<std::int32_t>(1 + rng() % 12),
                                  64 * static_cast<std::int32_t>(1 + rng() % 40)});
                shapes.push_back({static_cast<std::int32_t>(1 + rng() % 1500),
                                  64 * static_cast<std::int32_t>(1 + rng() % 40)});
            }
        } else if (f.qtype == QType::MXFP4) {
            // Any N; K a whole number of 32-value exponent groups (2880: gpt-oss's hidden).
            shapes = {{7, 32}, {24, 96}, {100, 2880}, {333, 1056}, {1, 64}, {1000, 160}};
            for (int i = 0; i < 3; ++i) {
                shapes.push_back({static_cast<std::int32_t>(1 + rng() % 1500),
                                  32 * static_cast<std::int32_t>(1 + rng() % 100)});
            }
        } else if (ggml_block32(f.qtype)) {
            // Any N; K whole 32-value blocks (5376: Gemma 4 31B's hidden; 1856: Nemotron-H's
            // expert FFN). 96, 1056, 2880 and 1856 end in a part step: blocks read block by block.
            shapes = {{7, 128}, {24, 96}, {100, 1024}, {333, 1056}, {1, 128}, {40, 2880},
                      {64, 5376}, {48, 1856}, {5, 32}};
            for (int i = 0; i < 3; ++i) {
                shapes.push_back({static_cast<std::int32_t>(1 + rng() % 1500),
                                  32 * static_cast<std::int32_t>(1 + rng() % 128)});
            }
        } else if (ggml_kquant(f.qtype)) {
            // Any N; K whole 256-value superblocks (1024: Qwen3.5-0.8B's hidden, 3584 its FFN).
            shapes = {{7, 256}, {24, 512}, {100, 1024}, {333, 1280}, {1, 256}, {40, 3584}};
            for (int i = 0; i < 3; ++i) {
                shapes.push_back({static_cast<std::int32_t>(1 + rng() % 1500),
                                  256 * static_cast<std::int32_t>(1 + rng() % 16)});
            }
        } else if (u4_zp_group(f.qtype)) {
            // Any N; K a whole number of groups (2560, 9728: Qwen3-4B's projections; 3840: Gemma 4
            // 12B's hidden; G32/G64 also K that the tiled route's 64-value tiles do not divide).
            const auto g = static_cast<std::int32_t>(u4_zp_group(f.qtype));
            shapes = {{7, 128}, {24, 256}, {100, 2560}, {333, 1152}, {1, 128}, {1000, 384},
                      {40, 9728}, {17, 3840}};
            if (g < 128) shapes.push_back({65, g == 32 ? 96 : 192});
            if (g == 32) shapes.push_back({9, 32});
            for (int i = 0; i < 3; ++i) {
                shapes.push_back({static_cast<std::int32_t>(1 + rng() % 1500),
                                  g * static_cast<std::int32_t>(1 + rng() % (3840 / g))});
            }
        } else {
            shapes = {{7, 40}, {24, 200}, {100, 1000}, {136, 1152}, {1, 64}, {1000, 72}, {333, 3072}};
            for (int i = 0; i < 4; ++i) {
                shapes.push_back({static_cast<std::int32_t>(1 + rng() % 1500),
                                  static_cast<std::int32_t>(8 + 8 * (rng() % 400))});
            }
            shapes.push_back({64, 100}); // K the register MMA cannot tile: scalar fallback
        }
        for (const auto& [n, k] : shapes) {
            for (const std::int32_t t : {1, 3, 8, 13, 17, 40, 130}) {
                for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                    failures += check_case(f, op, n, k, t, ops::LinearPolicy::A16Only,
                                           static_cast<std::uint32_t>(n * 131 + k * 7), nullptr,
                                           ActivationCompute::A16);
                }
            }
        }
    }

    // 2. NVFP4 W4A4 through the public entry points (AllowA4, T >= 17 takes it).
    const Format& nvfp4 = format_of(QType::NVFP4);
    std::vector<std::pair<std::int32_t, std::int32_t>> w4a4_shapes;
    if (kNativeNvfp4) { w4a4_shapes = {{384, 1088}, {1152, 960}, {256, 3840}, {200, 1088}}; }
    for (const auto& [n, k] : w4a4_shapes) {
        for (const std::int32_t t : {1, 16, 17, 64, 200}) {
            for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                failures += check_case(nvfp4, op, n, k, t, ops::LinearPolicy::AllowA4, 17U + t,
                                       nullptr,
                                       t >= 17 ? ActivationCompute::A4 : ActivationCompute::A16);
            }
        }
    }
    // 1c. A drafter's A16Only projection takes A8 exactly where R8 verifies an eight-token block
    // of its format (Q8 on every chip); another recorded policy stays.
    for (const Format& f : kFormats) {
        if (!on_build(f)) continue;
        const Weight w   = cached_weight(f.qtype, 333, 2560, 29U).device->weight;
        const bool block = generic::r8_verify_widest(f.qtype) >= generic::kR8VerifyTokens &&
                           generic::r8_takes_block(w);
        const ops::LinearPolicy want = block ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::A16Only;
        if (ops::drafter_linear_policy(w, ops::LinearPolicy::A16Only, ops::DrafterOp::Linear) != want ||
            ops::drafter_linear_policy(w, ops::LinearPolicy::AllowA4, ops::DrafterOp::Linear) !=
                ops::LinearPolicy::AllowA4 ||
            (f.qtype == QType::Q8_G32_FP16 && !block)) {
            std::cout << "FAIL " << f.name << " drafter_linear_policy" << std::endl;
            ++failures;
        }
    }
    // A Q8 shape the table of the op that runs it owns keeps its A16 kernels, since that
    // dispatcher sends only its one-token calls to R8; the segmented projection reads no table.
    // Gemma 4 31B DFlash's 5376 x 10752 (plain table), the Qwen DFlash down 2048 x 6144 (fused
    // add table only: lifted through plain linear), unowned shapes.
    for (const auto& [n, k] : std::initializer_list<std::pair<std::int32_t, std::int32_t>>{
             {5376, 10752}, {2048, 6144}, {4096, 4096}, {333, 2560}}) {
        const Weight w   = cached_weight(QType::Q8_G32_FP16, n, k, 31U).device->weight;
        const bool block = generic::r8_takes_block(w);
        const auto want  = [&](bool owned) {
            return !owned && block ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::A16Only;
        };
        const auto got = [&](ops::DrafterOp op) {
            return ops::drafter_linear_policy(w, ops::LinearPolicy::A16Only, op);
        };
        if (got(ops::DrafterOp::Linear) !=
                want(ops::detail::linear_shape_registered(QType::Q8_G32_FP16, n, k)) ||
            got(ops::DrafterOp::LinearAdd) != want(ops::detail::q8_linear_add_registered(n, k)) ||
            got(ops::DrafterOp::LinearSwiGlu) !=
                want(ops::detail::q8_linear_swiglu_registered(n, k)) ||
            got(ops::DrafterOp::Generic) != want(false) ||
            (n == 5376 && got(ops::DrafterOp::Linear) != ops::LinearPolicy::A16Only)) {
            std::cout << "FAIL Q8 " << n << " x " << k << " drafter_linear_policy" << std::endl;
            ++failures;
        }
    }

    // 2a. NVFP4 with A8 allowed and A4 not: W4A8 from 17 tokens, W4A4 kept under AllowA4.
    if (on_build(nvfp4)) failures += check_nvfp4_a8_routes();

    // 2b. W4A8 through the public entry points (AllowA8, T >= 17 takes it; K w4a8_k_admits: whole
    // tiles, or whole 32-value blocks for those formats, 2112 and 2880 among them), and R8
    // for the formats with an R8 codec at decode widths (kR8DecodeTokens) when the activation
    // fits the CTA, and at verify widths (r8_verify_widest) for r8_verify_takes formats. R8 needs
    // K only in whole load groups (r8_k_values): 2112 (Gemma 4 26B's dense down projection) and
    // 2880 (gpt-oss's hidden) take it where the layout allows them.
    // A format with an R8 codec alone (Q3_K) stays A16 from 17 tokens.
    for (const Format& f : kFormats) {
        if (!on_build(f)) continue;
        const bool w4a8 = generic::w4a8_supports(f.qtype);
        const bool r8   = generic::r8_supports(f.qtype);
        if (!w4a8 && !r8) continue;
        for (const auto& [n, k] : std::vector<std::pair<std::int32_t, std::int32_t>>{
                 {200, 1280}, {333, 2560}, {64, 5376}, {40, 12288}, {176, 2112}, {96, 2880}}) {
            if (k % f.k_multiple != 0) continue;
            const bool w4a8_k = w4a8 && generic::w4a8_k_admits(f.qtype, k);
            const bool r8_k   = r8 && k % generic::r8_k_values(f.qtype) == 0;
            for (const std::int32_t t : {1, 2, 3, 4, 8, 9, 16, 17, 64, 200}) {
                for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                    // The verify widths ask about the matrix itself (a gated op's weight holds
                    // both halves): a small Q4_0 one keeps A16 where the BF16 MMA is native.
                    const std::int64_t weight_rows = gated(op) ? 2 * n : n;
                    const bool verify              = t >= 2 && t <= generic::r8_verify_widest(f.qtype) &&
                                        generic::r8_verify_takes_shape(f.qtype, weight_rows, k);
                    const bool a8 =
                        (w4a8_k && t >= 17) ||
                        (r8_k &&
                         ((t <= generic::kR8DecodeTokens && generic::r8_fits(f.qtype, k, t)) ||
                          verify));
                    failures += check_case(f, op, n, k, t, ops::LinearPolicy::AllowA8,
                                           static_cast<std::uint32_t>(19U + t + k),
                                           nullptr,
                                           a8 ? ActivationCompute::A8 : ActivationCompute::A16);
                }
            }
        }
        // ggml's matrix kernel at a K of an odd number of 256-value steps (Gemma 4 31B 5376, 26B
        // 2816), forced: the quantizer's last CTA covers half a 512-value group.
        {
            const generic::Route mmq{generic::Family::GgmlMmq, 0};
            for (const std::int32_t k : {2816, 5376}) {
                if (k % f.k_multiple != 0) continue;
                const Weight probe = cached_weight(f.qtype, 128, k, 23U).device->weight;
                if (!generic::route_admits(mmq, probe, ops::LinearPolicy::AllowA8)) continue;
                for (const std::int32_t t : {9, 16, 33}) {
                    failures += check_case(f, Op::Linear, 128, k, t, ops::LinearPolicy::AllowA8,
                                           23U, &mmq, ActivationCompute::A8);
                }
            }
        }
        // Every R8 schedule, every epilogue, every token-slot width; then a token's bytes do not
        // depend on T.
        for (int i = 0; r8 && i < generic::r8_schedule_count(); ++i) {
            const generic::Route r{generic::Family::R8Mma, static_cast<std::uint8_t>(i)};
            for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                for (const std::int32_t t : {1, 2, 3, 4}) {
                    failures += check_case(f, op, 333, 2560, t, ops::LinearPolicy::AllowA8,
                                           13U + i, &r, ActivationCompute::A8);
                }
            }
            failures += check_r8_invariance(f, Op::Linear, 136, 2560, r, 67U);
            failures += check_r8_invariance(f, Op::SwiGlu, 136, 2560, r, 71U);
            if (2112 % f.k_multiple == 0 && 2112 % generic::r8_k_values(f.qtype) == 0) {
                failures += check_r8_invariance(f, Op::Linear, 176, 2112, r, 73U);
            }
        }
        if (r8) {
            failures += check_r8_schedules_agree(f, Op::Linear, 141, 12288, 89U);
            failures += check_r8_schedules_agree(f, Op::Add, 141, 5376, 93U);
            failures += check_r8_schedules_agree(f, Op::SwiGlu, 141, 5376, 97U);
        }
        // The automatic W4A8 pick on a grid of 128-token tiles that fills the GPU (T = 1024, 1536
        // rows: 96 CTAs of 128x128, the RTX 4090's 3/4-of-the-SMs threshold, 192 for SwiGLU):
        // where the RTX 4090 takes Q4_K's 128x128 tile (w4a8_wide_q4k) and every format's 128-token
        // tile runs the token-major grid, through the public entry points against the oracle.
        // The pick's own bounds follow the device's SM count: at T = 1024 (8 token tiles) the
        // smallest row count whose CTAs reach 3/4 of the SMs takes 128x128, one 128-row block
        // fewer does not (the RTX 4090's 1536 and 1408; an 80-SM RTX 4080 SUPER's 1024 and 896).
        const auto wide_rows = [](bool under) {
            const std::int64_t sms    = device_class().sms;
            const std::int64_t blocks = (3 * sms + 31) / 32 - (under ? 1 : 0);
            return static_cast<std::int32_t>(128 * std::max<std::int64_t>(blocks, 1));
        };
        if (w4a8) {
            const auto& d  = device_class();
            const bool ada = d.major == 8 && d.minor == 9;
            for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                failures += check_case(f, op, 1536, 256, 1024, ops::LinearPolicy::AllowA8, 29U,
                                       nullptr, ActivationCompute::A8);
            }
            if (ada && f.qtype == QType::GGML_Q4_K) {
                const HostWeight host = make_weight(f.qtype, 1536, 256, 29U);
                DeviceWeight dw       = upload_weight(host);
                const generic::Route wide{generic::Family::W4a8Mma, 2}; // 128x128
                for (const auto epi : {generic::Epilogue::Store, generic::Epilogue::AddResidual,
                                       generic::Epilogue::SwiGlu}) {
                    const auto a8 = ops::LinearPolicy::AllowA8;
                    // 1536 rows at T = 1024 (96 CTAs on the RTX 4090) take it; 256 rows at
                    // T = 200 (4) do not.
                    const generic::Problem p{f.qtype, a8, wide_rows(false), 256, epi};
                    const generic::Route big = generic::w4a8_route(p, dw.weight, 1024).value();
                    const generic::Problem small{f.qtype, a8, 256, 256, epi};
                    const generic::Route few = generic::w4a8_route(small, dw.weight, 200).value();
                    if (!(big == wide) || few == wide) {
                        std::cerr << "q4k W4A8 pick: " << generic::route_name(big) << " at "
                                  << wide_rows(false) << " x T 1024, " << generic::route_name(few)
                                  << " at 256 x T 200\n";
                        ++failures;
                    }
                }
                std::cout << "q4k W4A8 pick on sm_89 (" << d.sms << " SMs): 128x128 from "
                          << wide_rows(false) / 16 << " CTAs, not below\n";
            }
        }
        // Pick only (the W4A8 rule a device without clocks runs, w4a8_route; the cost model's picks
        // are the selector golden test's), at the bounds of the sm_89
        // gates: Q4_0 takes the producer-warp 128x64 tile (5) on sm_89 from 5376 x 5376 and from
        // 128 tokens, on Ampere (sm_80/86) at every shape from 128 tokens, and 64x64
        // (4) below either bound and on every other GPU; Q4_K one 128-row
        // block under the device's 3/4-of-the-SMs grid at T = 1024 (1408 rows, 88 CTAs, on the
        // RTX 4090) does not take 128x128 (2).
        if (w4a8 && (f.qtype == QType::GGML_Q4_0 || f.qtype == QType::GGML_Q4_K)) {
            const auto& d  = device_class();
            const bool ada    = d.major == 8 && d.minor == 9;
            const bool ampere = d.major == 8 && d.minor < 9;
            const auto a8     = ops::LinearPolicy::AllowA8;
            const auto pick = [&](std::int32_t n, std::int32_t k, std::int32_t t,
                                  generic::Epilogue epi) {
                const DeviceWeight dw = upload_weight(make_weight(f.qtype, n, k, 31U));
                return generic::w4a8_route(generic::Problem{f.qtype, a8, n, k, epi}, dw.weight, t)
                    .value();
            };
            const auto expect = [&](bool ok, const char* what, generic::Route got) {
                if (ok) return;
                std::cerr << f.name << " W4A8 pick " << what << ": " << generic::route_name(got)
                          << "\n";
                ++failures;
            };
            const generic::Route ws{generic::Family::W4a8Mma, 5};
            const generic::Route small{generic::Family::W4a8Mma, 4};
            const generic::Route wide{generic::Family::W4a8Mma, 2};
            if (f.qtype == QType::GGML_Q4_0) {
                for (const auto epi : {generic::Epilogue::Store, generic::Epilogue::AddResidual,
                                       generic::Epilogue::SwiGlu}) {
                    const generic::Route at = pick(5376, 5376, 1024, epi);
                    expect(at == (ada || ampere ? ws : small), "5376 x 5376, T 1024", at);
                    const generic::Route short_k = pick(5376, 2816, 1024, epi);
                    expect(short_k == (ampere ? ws : small), "5376 x 2816, T 1024", short_k);
                    const generic::Route few_rows = pick(2816, 5376, 1024, epi);
                    expect(few_rows == (ampere ? ws : small), "2816 x 5376, T 1024", few_rows);
                    const generic::Route narrow = pick(5376, 5376, 127, epi);
                    expect(narrow == small, "5376 x 5376, T 127", narrow);
                }
            } else {
                const generic::Route under =
                    pick(wide_rows(true), 256, 1024, generic::Epilogue::Store);
                expect(!(under == wide), "one block under 3/4 of the SMs, T 1024", under);
            }
            std::cout << f.name << " W4A8 pick bounds checked" << (ada ? " on sm_89" : "")
                      << "\n";
        }
        // Every W4A8 schedule this build's shared memory holds, every epilogue.
        for (int i = 0; w4a8 && i < generic::w4a8_schedule_count(); ++i) {
            if (!generic::w4a8_fits(i, f.qtype)) continue;
            const generic::Route r{generic::Family::W4a8Mma, static_cast<std::uint8_t>(i)};
            for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                for (const std::int32_t t : {1, 37, 200}) {
                    failures += check_case(f, op, 200, 1280, t, ops::LinearPolicy::AllowA8,
                                           11U + i, &r, ActivationCompute::A8);
                }
            }
        }
    }

    // 2b'. I8 (Q4 / Q5 / Q6 G64, one activation scale per 64 values): every schedule, every epilogue, a row
    // count that is no tile multiple and T across tile edges, against the oracle of x quantized
    // per 64; and every token byte-identical across schedules, call widths and starts.
    for (const Format& f : kFormats) {
        const bool g64 = f.qtype == QType::Q4_G64_FP16 || f.qtype == QType::Q5_G64_FP16 ||
                         f.qtype == QType::Q6_G64_FP16;
        if (!on_build(f) || !g64 || !ninferno::device_class().features.mma_bf16_native) {
            continue;
        }
        std::vector<generic::Route> i8;
        for (int i = 0; i < generic::i8_schedule_count(); ++i) {
            if (generic::i8_fits(i, f.qtype)) {
                i8.push_back({generic::Family::I8Mma, static_cast<std::uint8_t>(i)});
            }
        }
        for (const generic::Route& r : i8) {
            for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                for (const std::int32_t t : {17, 200, 300}) {
                    failures += check_case(f, op, 200, 1536, t, ops::LinearPolicy::AllowA8,
                                           91U + r.schedule, &r, ActivationCompute::A8);
                }
            }
        }
        for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
            failures += check_invariance(f, op, 136, 1536, ops::LinearPolicy::AllowA8, 97U, &i8);
        }
    }

    // 2c. The per-GPU R8 decode entries name compiled schedules of formats with an R8 codec at
    // problems R8 takes at T = 1 (whose schedules agree byte for byte, above), one table per GPU
    // class id and one entry per problem in it.
    std::set<std::string_view> gpus;
    for (const generic::R8DecodeTable& table : generic::kR8DecodeTables) {
        if (!gpus.insert(table.gpu).second) {
            std::cerr << "R8 decode tables: two for " << table.gpu << "\n";
            ++failures;
        }
        std::set<std::tuple<int, int, std::int32_t, std::int32_t>> problems;
        for (const generic::R8DecodeEntry& e : table.entries) {
            if (!problems
                     .insert({static_cast<int>(e.qtype), static_cast<int>(e.epilogue), e.n, e.k})
                     .second) {
                std::cerr << "R8 decode table " << table.gpu << ": two entries for n=" << e.n
                          << " k=" << e.k << "\n";
                ++failures;
            }
            bool compiled = false;
            for (int i = 0; i < generic::r8_schedule_count(); ++i) {
                const auto s = generic::r8_schedule_info(i);
                compiled     = compiled || (s.warps == e.warps && s.rows == e.rows);
            }
            if (!compiled || !generic::r8_supports(e.qtype) || (e.k % 128) != 0 ||
                !generic::r8_fits(e.qtype, e.k, 1) ||
                (e.epilogue != generic::Epilogue::Store && e.epilogue != generic::Epilogue::SwiGlu)) {
                std::cerr << "R8 decode table " << table.gpu << ": bad entry n=" << e.n
                          << " k=" << e.k << "\n";
                ++failures;
            }
        }
    }

    // 2c. ggml's matrix kernel (Family::GgmlMmq) for the formats it reads, every epilogue, at
    // decode, verify and prefill widths, under the A8 allowance against the unquantized oracle.
    {
        const generic::Route mmq{generic::Family::GgmlMmq, 0};
        for (const Format& f : kFormats) {
            if (!on_build(f)) continue;
            if (!generic::ggml_mmq_format(f.qtype)) continue;
            for (const auto& [n, k] : std::vector<std::pair<std::int32_t, std::int32_t>>{
                     {128, 512}, {384, 2560}, {640, 5120}}) {
                if (!generic::route_admits(mmq, cached_weight(f.qtype, n, k, 5U).device->weight,
                                           ops::LinearPolicy::AllowA8)) {
                    std::cout << f.name << " ggml_mmq: not admitted on this device\n";
                    break;
                }
                // 600: a prefill call in two slices of the route's 512 tokens, the second short.
                for (const std::int32_t t : {1, 2, 4, 5, 8, 16, 17, 24, 64, 130, 600}) {
                    for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                        failures += check_case(f, op, n, k, t, ops::LinearPolicy::AllowA8,
                                               static_cast<std::uint32_t>(29U + t + k), &mmq,
                                               ActivationCompute::A8);
                    }
                }
            }
        }
    }

    // 2d. ggml's vector kernel (Family::GgmlVec) likewise at 1..8 tokens.
    {
        const generic::Route vec{generic::Family::GgmlVec, 0};
        for (const Format& f : kFormats) {
            if (!on_build(f)) continue;
            if (!generic::ggml_vec_format(f.qtype)) continue;
            std::vector<std::pair<std::int32_t, std::int32_t>> shapes{
                {128, 512}, {200, 2560}, {640, 5120}};
            // K 2880 (gpt-oss): whole blocks of the 32-value formats, the slice loop's tail.
            if (ggml_block_values(f.qtype) < 256) shapes.push_back({96, 2880});
            for (const auto& [n, k] : shapes) {
                if (!generic::route_admits(vec, cached_weight(f.qtype, n, k, 5U).device->weight,
                                           ops::LinearPolicy::AllowA8)) {
                    std::cout << f.name << " ggml_vec: not admitted\n";
                    break;
                }
                for (const std::int32_t t : {1, 2, 3, 4, 5, 8}) {
                    for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                        failures += check_case(f, op, n, k, t, ops::LinearPolicy::AllowA8,
                                               static_cast<std::uint32_t>(31U + t + k), &vec,
                                               ActivationCompute::A8);
                    }
                }
            }
        }
    }

    // 2c'. A prefill chunk past the 65535 tokens of a grid's y extent: every token's epilogue runs.
    // Each SwiGLU half is a whole 128-row tile, as the kernel takes it (a 64-row half runs A16).
    {
        const generic::Route mmq{generic::Family::GgmlMmq, 0};
        for (const Format& f : kFormats) {
            if (f.qtype != QType::GGML_Q4_K || !on_build(f)) continue;
            if (!generic::route_admits(mmq, cached_weight(f.qtype, 128, 256, 41U).device->weight,
                                       ops::LinearPolicy::AllowA8)) {
                continue;
            }
            failures += check_case(f, Op::SwiGlu, 128, 256, 65600, ops::LinearPolicy::AllowA8, 41U,
                                   &mmq, ActivationCompute::A8);
        }
    }

    // 2e. A q8_1 block whose sum passes fp16's range stays finite through ggml's matrix kernel.
    for (const Format& f : kFormats) {
        if (!on_build(f) || !generic::ggml_mmq_format(f.qtype)) continue;
        failures += check_mmq_outlier_block(f);
    }
    // 2f. The q8_1 block sums from the codes: ggml's matrix kernel on the K-quants against the int8
    // activation's oracle.
    for (const Format& f : kFormats) {
        const bool k_quant = f.qtype == QType::GGML_Q4_K || f.qtype == QType::GGML_Q5_K;
        if (!on_build(f) || !k_quant || !generic::ggml_mmq_format(f.qtype)) continue;
        failures += check_mmq_code_sums(f);
    }

    // 3. Every compiled route qualified on one awkward shape per format and epilogue.
    for (const Format& f : kFormats) {
        if (!on_build(f)) continue;
        const std::int32_t n = 200; // NVFP4 too: a last scale tile its rows do not fill
        const std::int32_t k = awkward_k(f);
        for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
            const generic::Route r{generic::Family::RegisterMma, static_cast<std::uint8_t>(i)};
            if (!rmma_compiled(f.qtype, i, k)) continue;
            for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                for (const std::int32_t t : {1, 37}) {
                    failures += check_case(f, op, n, k, t, ops::LinearPolicy::A16Only, 5U + i, &r,
                                           ActivationCompute::A16);
                }
            }
        }
        const generic::Route scalar{generic::Family::Scalar, 0};
        failures += check_case(f, Op::SwiGlu, n, k, 9, ops::LinearPolicy::A16Only, 3U, &scalar,
                               ActivationCompute::A16);
        if (!tiled_format(f.qtype) || (k % 64) != 0) continue;
        Weight format{};
        format.qtype = f.qtype;
        for (int i = 0; i < generic::tiled_schedule_count(); ++i) {
            // The schedules this build's shared memory holds (not the 128x128 tile on sm_75).
            if (!generic::tiled_fits(i, generic::codec_of(format))) continue;
            const generic::Route r{generic::Family::TiledMma, static_cast<std::uint8_t>(i)};
            for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                for (const std::int32_t t : {1, 37, 200}) {
                    failures += check_case(f, op, n, k, t, ops::LinearPolicy::A16Only, 7U + i, &r,
                                           ActivationCompute::A16);
                }
            }
        }
    }
    for (int i = 0; kNativeNvfp4 && i < generic::w4a4_schedule_count(); ++i) {
        const generic::Route r{generic::Family::W4a4Mma, static_cast<std::uint8_t>(i)};
        for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
            for (const std::int32_t n : {384, 200}) {
                failures += check_case(nvfp4, op, n, 1088, 77, ops::LinearPolicy::AllowA4, 9U + i,
                                       &r, ActivationCompute::A4);
            }
        }
    }
    // W4A4 route names parse on every build (a shipped Blackwell table names them); a build without
    // the NVFP4 MMA refuses them in admission and on the explicit-route path.
    {
        const auto route = generic::parse_route("w4a4.32x64");
        if (!route || route->family != generic::Family::W4a4Mma) {
            std::cerr << "FAIL w4a4.32x64 does not parse as a W4A4 route\n";
            ++failures;
        } else {
            const HostWeight host = make_weight(QType::NVFP4, 256, 1024, 41U);
            DeviceWeight device   = upload_weight(host);
            const bool admitted =
                generic::route_admits(*route, device.weight, ops::LinearPolicy::AllowA4);
            if (admitted != kNativeNvfp4) {
                std::cerr << "FAIL w4a4.32x64 admitted=" << admitted << " on a build with"
                          << (kNativeNvfp4 ? "" : "out") << " the NVFP4 MMA\n";
                ++failures;
            }
            if (!kNativeNvfp4) {
                DeviceBuffer x(1024 * 2), out(256 * 2);
                Tensor tx(x.p, DType::BF16, {1024, 1});
                Tensor tout(out.p, DType::BF16, {256, 1});
                DeviceArena workspace(1U << 20);
                generic::Call call;
                call.x               = &tx;
                call.weight          = &device.weight;
                call.rows            = 256;
                call.epilogue        = generic::Epilogue::Store;
                call.out[0]          = &tout;
                call.segment_rows[0] = 256;
                bool refused         = false;
                try {
                    generic::run_route(*route, call, &workspace, nullptr);
                } catch (const std::invalid_argument&) { refused = true; }
                if (!refused) {
                    std::cerr << "FAIL an explicit W4A4 route ran on a build without it\n";
                    ++failures;
                }
            }
        }
    }
    // Every staged schedule (BF16, K a multiple of 64), with partial row and token tiles.
    const Format& bf16 = format_of(QType::BF16);
    for (int i = 0; i < generic::staged_schedule_count(); ++i) {
        // The schedules this build's shared memory holds (not the 128x128 tile on sm_75).
        if (!generic::staged_fits(i)) continue;
        const generic::Route r{generic::Family::StagedMma, static_cast<std::uint8_t>(i)};
        for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
            for (const std::int32_t t : {1, 37, 200}) {
                failures += check_case(bf16, op, 200, 1088, t, ops::LinearPolicy::A16Only, 11U + i,
                                       &r, ActivationCompute::A16);
            }
        }
    }
    for (int i = 0; kNativeNvfp4 && i < generic::nvfp4_tiled_schedule_count(); ++i) {
        const generic::Route r{generic::Family::Nvfp4TiledMma, static_cast<std::uint8_t>(i)};
        for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
            for (const std::int32_t t : {1, 37, 200}) {
                for (const std::int32_t n : {384, 200}) {
                    failures += check_case(nvfp4, op, n, 1088, t, ops::LinearPolicy::A16Only,
                                           13U + i, &r, ActivationCompute::A16);
                }
            }
        }
    }

    // 3a. u4 G32/G64 at K in whole 128-value steps take the quad-run spread decoder (awkward_k
    // above takes the one-group-per-step one): every register-MMA schedule.
    for (const Format& f : kFormats) {
        if (!on_build(f)) continue;
        if (u4_zp_group(f.qtype) != 32 && u4_zp_group(f.qtype) != 64) continue;
        for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
            const generic::Route r{generic::Family::RegisterMma, static_cast<std::uint8_t>(i)};
            if (!rmma_compiled(f.qtype, i, 1152)) continue;
            for (const Op op : {Op::Linear, Op::SwiGlu, Op::GeGlu}) {
                for (const std::int32_t t : {1, 37}) {
                    failures += check_case(f, op, 200, 1152, t, ops::LinearPolicy::A16Only, 71U + i,
                                           &r, ActivationCompute::A16);
                }
            }
        }
    }

    // 3b. Act-order u4 weights (RowGroupZpOrdered): the public entry points and every route family
    // gather the input into the stored column order first.
    for (const Format& f : kFormats) {
        if (!on_build(f)) continue;
        if (!u4_zp_group(f.qtype)) continue;
        const std::int32_t k = awkward_k(f);
        for (const std::int32_t t : {1, 5, 40, 300}) {
            for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                failures += check_case(f, op, 200, k, t, ops::LinearPolicy::A16Only, 61U + t,
                                       nullptr, ActivationCompute::A16, true);
            }
        }
        const generic::Route routes[] = {{generic::Family::Scalar, 0},
                                         {generic::Family::RegisterMma, 0},
                                         {generic::Family::TiledMma, 0}};
        for (const auto& r : routes) {
            failures += check_case(f, Op::SwiGlu, 136, k, 37, ops::LinearPolicy::A16Only, 67U, &r,
                                   ActivationCompute::A16, true);
        }
    }
    // 3c. The NVFP4 tiled routes sum in the register MMA's order: K of 17 steps (uneven classes),
    // 4 steps (empty classes) and one step.
    for (const std::int32_t k : {1088, 256, 64}) {
        for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
            if (!kNativeNvfp4) break;
            failures += check_nvfp4_tiled_order(nvfp4, op, k, 83U + k);
        }
    }

    // 4. Token invariance of the prefill-admissible routes.
    for (const Format& f : kFormats) {
        if (!on_build(f)) continue;
        const std::int32_t n = f.qtype == QType::NVFP4 ? 256 : 136;
        const std::int32_t k = f.qtype == QType::NVFP4 ? 1024 : awkward_k(f);
        failures += check_invariance(f, Op::Linear, n, k, ops::LinearPolicy::A16Only, 41U);
        failures += check_invariance(f, Op::SwiGlu, n, k, ops::LinearPolicy::A16Only, 43U);
    }
    // BF16 at a K the staged schedules admit: they join the register-MMA routes' bytes.
    failures += check_invariance(bf16, Op::Linear, 136, 1088, ops::LinearPolicy::A16Only, 59U);
    failures += check_invariance(bf16, Op::SwiGlu, 136, 1088, ops::LinearPolicy::A16Only, 61U);
    if (kNativeNvfp4) {
        failures += check_invariance(nvfp4, Op::Linear, 256, 1024, ops::LinearPolicy::AllowA4, 47U);
        failures += check_invariance(nvfp4, Op::SwiGlu, 256, 1024, ops::LinearPolicy::AllowA4, 53U);
    }
    // W4A8: every schedule each format is admitted to (the classic tiles and, for five formats,
    // the producer-warp ring of three stages) at K of ten 128-value tiles and at K of six, four,
    // three, two and one: twice the ring, one more tile than stages, as many (three and one only
    // for 32/128-value block formats, the ring's Q4_0 among them), fewer; with each epilogue.
    for (const Format& f : kFormats) {
        if (!on_build(f)) continue;
        if (!generic::w4a8_supports(f.qtype)) continue;
        failures += check_invariance(f, Op::Linear, 136, 1280, ops::LinearPolicy::AllowA8, 59U);
        failures += check_invariance(f, Op::SwiGlu, 136, 1280, ops::LinearPolicy::AllowA8, 61U);
        failures += check_invariance(f, Op::Add, 136, 1280, ops::LinearPolicy::AllowA8, 63U);
        for (const std::int32_t k : {768, 512, 384, 256, 128}) {
            if (k % f.k_multiple != 0) continue;
            for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                failures += check_invariance(f, op, 136, k, ops::LinearPolicy::AllowA8, 73U + k);
            }
        }
    }

    for (const Format& f : kFormats) {
        if (on_build(f) && f.qtype == QType::GGML_IQ3_XXS) {
            failures += check_w4a8_misaligned_signs(f);
        }
    }

    // 4b. Volta's tensor-core verify route: token invariance over 2..16 (sm_70 only).
    if (!device_class().features.mma_tensor) {
        for (const Format& f : kFormats) {
            if (!on_build(f) || (f.qtype != QType::GGML_Q4_K && f.qtype != QType::GGML_Q5_K &&
                                 f.qtype != QType::GGML_Q6_K && f.qtype != QType::Q8_G32_FP16)) {
                continue;
            }
            for (const Op op : {Op::Linear, Op::Add, Op::SwiGlu, Op::GeGlu, Op::Relu2}) {
                failures +=
                    check_volta_hmma_invariance(f, op, 72, 512, ops::LinearPolicy::A16Only, 79U);
            }
            failures +=
                check_volta_hmma_invariance(f, Op::Add, 72, 512, ops::LinearPolicy::AllowA8, 81U);
            failures += check_volta_hmma_invariance(f, Op::SwiGlu, 40, 768,
                                                    ops::LinearPolicy::A16Only, 83U);
        }
    }

    // 5. The invariant route profile: each K split (8, 4 and 1 warps) on every format.
    for (const Format& f : kFormats) {
        if (!on_build(f)) continue;
        const std::int32_t n = f.qtype == QType::NVFP4 ? 256 : 136;
        for (const std::int32_t k : {awkward_k(f), 768, 256}) {
            failures += check_invariant_profile(f, Op::Linear, n, k, 67U);
            failures += check_invariant_profile(f, Op::SwiGlu, n, k, 71U);
        }
    }

    if (failures != 0) {
        std::cerr << "generic linear: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "generic linear: all cases passed\n";
    return 0;
}
