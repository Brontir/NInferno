// The dense pass's side-by-side projections (DensePass::project_side_by_side in
// models/dense/target.cu) run a layer's q/k/v and attention gate as one segmented projection:
// each call after the first joins the one before it (generic Call::join_previous), and a thin
// decode call may be spread over the SMs (GemmArgs::spread_smem). Neither may change a byte:
// wherever the dense pass takes them (ops::detail::side_by_side_admits, the one answer both
// read), every output must equal what the separate projections (DensePass::project ->
// ops::linear) give, bitwise. Every format the generic register MMA reads is offered, so a format
// the predicate starts to admit is checked here without a new case; under the speed profile and
// under the invariant one (--activations exact), at decode and verify widths.
#include "core/device_class.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/segmented_projection.h"

#include "core/arena.h"
#include "ops/linear/linear_registry.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninferno;
using namespace ninferno::test;
using namespace ninferno::test::linear;

namespace {

struct Slice {
    QType qtype;
    std::int32_t rows;
};

quantized_weight::PackedWeight make(QType qtype, std::int32_t n, std::int32_t k,
                                    std::uint32_t seed) {
    if (qtype == QType::NVFP4) return make_nvfp4_weight(n, k, seed);
    if (qtype == QType::Q8_G32_FP16) return make_q8_g32_fp16_weight(n, k, seed);
    if (qtype == QType::Q4_G64_FP16) return make_q4_g64_fp16_weight(n, k, seed);
    if (qtype == QType::Q6_G64_FP16) return make_q6_g64_fp16_weight(n, k, seed);
    return quantized_weight::make_patterned_weight(qtype, n, k, seed);
}

const char* name(QType q) {
    return q == QType::NVFP4                ? "nvfp4"
           : q == QType::FP8_E4M3FN_ROW_F32 ? "fp8_row_f32"
           : q == QType::Q8_G32_FP16        ? "q8_g32"
           : q == QType::Q4_G64_FP16        ? "q4_g64"
           : q == QType::Q6_G64_FP16        ? "q6_g64"
           : q == QType::GGML_Q4_K          ? "q4_k"
           : q == QType::GGML_Q5_K          ? "q5_k"
           : q == QType::GGML_Q6_K          ? "q6_k"
           : q == QType::GGML_IQ4_XS        ? "iq4_xs"
           : q == QType::GGML_Q4_0          ? "q4_0"
                                            : "fp8_row_bf16";
}

// One layer's projections of x [K, T]: separately, then side by side; the outputs must match.
int check(const std::vector<Slice>& slices, std::int32_t k, std::int32_t t,
          ops::LinearPolicy policy, std::uint32_t seed, int& admitted) {
    std::string label = "[";
    for (std::size_t i = 0; i < slices.size(); ++i) {
        label += (i ? " " : "") + std::string(name(slices[i].qtype)) + ":" +
                 std::to_string(slices[i].rows);
    }
    label += std::string("] ") + (ops::invariant_routes_active() ? "exact " : "") +
             "K=" + std::to_string(k) + " T=" + std::to_string(t) +
             (policy == ops::LinearPolicy::AllowA4   ? " A4"
              : policy == ops::LinearPolicy::AllowA8 ? " A8"
                                                     : "");

    std::vector<quantized_weight::PackedWeight> host;
    std::vector<DeviceBuffer> payload;
    std::vector<Weight> weights;
    for (std::size_t i = 0; i < slices.size(); ++i) {
        const std::uint32_t s = seed + 17U * static_cast<std::uint32_t>(i);
        host.push_back(make(slices[i].qtype, slices[i].rows, k, s));
        payload.push_back(to_device(host.back().payload));
    }
    for (std::size_t i = 0; i < slices.size(); ++i) {
        weights.push_back(host[i].device_weight(payload[i].p));
    }
    // What the dense pass would do with these weights: join them only where every one admits it.
    for (std::size_t i = 0; i < slices.size(); ++i) {
        if (!ops::detail::side_by_side_admits(weights[i], policy, t)) {
            std::cout << "refused " << label << ": output " << i << "\n";
            return 0;
        }
    }
    ++admitted;
    const auto activation = make_linear_activation(k, t, seed + 7U, ActivationCompute::A16);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    const Tensor tx(x.p, DType::BF16, {k, t});

    std::vector<DeviceBuffer> separate, joined;
    std::vector<Tensor> ts, tj;
    for (const auto& s : slices) {
        separate.emplace_back(static_cast<std::size_t>(s.rows) * t * 2);
        joined.emplace_back(static_cast<std::size_t>(s.rows) * t * 2);
        cuda_check(cudaMemset(separate.back().p, 0xff, separate.back().bytes), "poison");
        cuda_check(cudaMemset(joined.back().p, 0x7f, joined.back().bytes), "poison");
    }
    for (std::size_t i = 0; i < slices.size(); ++i) {
        ts.push_back(Tensor(separate[i].p, DType::BF16, {slices[i].rows, t}));
        tj.push_back(Tensor(joined[i].p, DType::BF16, {slices[i].rows, t}));
    }

    ops::SegmentedProjectionWeights segmented;
    std::vector<ops::ProjectionOutput> outputs;
    std::size_t capacity = 256;
    for (std::size_t i = 0; i < slices.size(); ++i) {
        segmented.slices.push_back({weights[i], 0, slices[i].rows, policy});
        outputs.push_back({&tj[i], 0, slices[i].rows});
        capacity = std::max(capacity, ops::linear_workspace_capacity_bytes(
                                          slices[i].qtype, slices[i].rows, k, policy, t, t));
    }
    capacity = std::max(capacity,
                        ops::segmented_projection_workspace_capacity_bytes(segmented, t, t));
    DeviceArena workspace(capacity);

    for (std::size_t i = 0; i < slices.size(); ++i) {
        ops::linear(tx, weights[i], ts[i], policy, workspace, nullptr);
    }
    ops::segmented_projection(tx, segmented, outputs, workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "run");

    for (std::size_t i = 0; i < slices.size(); ++i) {
        const std::size_t count = static_cast<std::size_t>(slices[i].rows) * t;
        const auto a            = from_device<std::uint16_t>(separate[i], count);
        const auto b            = from_device<std::uint16_t>(joined[i], count);
        if (std::memcmp(a.data(), b.data(), count * 2) != 0) {
            std::size_t first = 0;
            while (a[first] == b[first]) ++first;
            std::cerr << "FAIL " << label << ": output " << i << " differs first at element "
                      << first << " (row " << first % slices[i].rows << ", token "
                      << first / slices[i].rows << "): 0x" << std::hex << a[first] << " vs 0x"
                      << b[first] << std::dec << "\n";
            return 1;
        }
    }
    std::cout << "ok " << label << "\n";
    return 0;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP: no CUDA device\n";
        return 77;
    }
    const QType nv  = QType::NVFP4;
    const QType f32 = QType::FP8_E4M3FN_ROW_F32;
    const QType b16 = QType::FP8_E4M3FN_ROW_BF16;
    const QType q8  = QType::Q8_G32_FP16;
    const QType q4  = QType::Q4_G64_FP16;
    const QType q6  = QType::Q6_G64_FP16;
    const QType k4  = QType::GGML_Q4_K;
    const QType k5  = QType::GGML_Q5_K;
    const QType k6  = QType::GGML_Q6_K;
    const QType i4  = QType::GGML_IQ4_XS;
    const QType l4  = QType::GGML_Q4_0;
    // q, k, v and the gate (Muse Glimmer 30B's are 4096, 256 x 2 and 4096 rows at K = 6656,
    // scaled down), and a packed q/k/v beside its gate.
    const std::vector<std::vector<Slice>> layers = {
        {{nv, 1024}, {nv, 256}, {nv, 256}, {nv, 1024}},
        {{f32, 1024}, {f32, 256}, {b16, 256}, {f32, 1024}},
        {{nv, 1024}, {nv, 256}, {nv, 256}, {f32, 1024}},
        {{f32, 1024}, {nv, 256}, {b16, 256}, {nv, 1024}},
        {{nv, 1536}, {nv, 1024}},
        {{f32, 1536}, {f32, 1024}},
        {{q8, 1536}, {q8, 768}},
        {{q8, 1536}, {nv, 256}, {f32, 256}, {q8, 768}},
        {{q8, 1024}, {q8, 256}},
        {{q4, 1536}, {q4, 768}},
        {{q6, 1536}, {nv, 256}, {q6, 768}},
        // GGUF layers on the R8 decode kernel: a packed [q; k] (or [q; k; gate]) beside a wider v.
        {{k4, 1280}, {k6, 256}},
        {{k4, 1024}, {k4, 256}, {k6, 256}, {k4, 1024}},
        {{k5, 1536}, {k6, 512}},
        {{i4, 1024}, {k4, 256}, {k6, 256}},
        {{l4, 1536}, {l4, 768}},
    };
    int failures = 0;
    int admitted = 0;
    for (const bool exact : {false, true}) {
        const ops::ScopedInvariantRoutes profile(exact);
        std::uint32_t seed = 101;
        for (const auto& layer : layers) {
            // A layer with an NVFP4 slice runs only in the 120a build; its seeds are still drawn,
            // so the other layers see the same weights in every build.
            const bool runs = std::ranges::all_of(layer, [](const Slice& slice) {
                return qtype_runs(slice.qtype, "side-by-side layer");
            });
            if (!runs) {
                seed += 12;
                continue;
            }
            for (const std::int32_t t : {1, 3, 8, 16}) {
                for (const auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8,
                                          ops::LinearPolicy::AllowA4}) {
                    failures += check(layer, 2048, t, policy, seed++, admitted);
                }
            }
        }
    }
    // A predicate that refused everything would pass vacuously.
    // Side by side needs programmatic dependent launch to join (side_by_side_admits): a device
    // without it admits nothing, and every projection runs apart.
    if (!ninferno::device_class().features.pdl) {
        if (admitted != 0) {
            std::cerr << "FAIL a device without PDL admitted " << admitted << " cases\n";
            ++failures;
        }
    } else if (admitted == 0) {
        std::cerr << "FAIL no case was admitted side by side\n";
        ++failures;
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " side-by-side projections, " << admitted
              << " admitted, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
