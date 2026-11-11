// The register MMA's pre-summed activation group plane (#619, GemmArgs::group_sums). Host: which
// calls take it (rmma_group_sums: the min-term formats on tensor-core MMA in the prefill profile
// at every width, never in the speed or invariant profile; rmma_plane_taken: only where the
// board's committed kernel facts admit the format and epilogue, every plane instance keeping its
// twin's spill and CTAs per SM, re-derived here from the snapshot rows), its row length and
// workspace, the workspace a plan reserves for it, and resolve_route's demotion of a call without
// room for it (the first reason of a call demoted before stands). GPU (skipped without one): the
// pre-pass equals the CPU's tree sum bit for bit for every K, T, group size, exponent spread and
// alignment, writes zero past K and nothing for tokens past the call's; the price describes the
// instances with the MMA with ones or the adds in every profile (the kernel-facts key's `sums`),
// the plane's instance a call takes agrees with the GPU's binary, and a captured prefill call
// replays as run eagerly.
#include "ops/quantized_weight.h"

#include "core/arena.h"
#include "core/device_class.h"
#include "core/kernel_occupancy.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/kernel_facts.h"
#include "ops/generic/route_query.h"
#include "ops/generic/route_table.h"
#include "ops/generic/selector_facts.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
namespace generic = ninferno::ops::detail::generic;
namespace kf      = generic::kernel_facts;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// The pre-pass's order (generic_group_sums.cu rmma_group_sum): the two halves' sums added.
float tree_sum(const float* v, int n) {
    if (n == 1) return v[0];
    return tree_sum(v, n / 2) + tree_sum(v + n / 2, n / 2);
}

float bf16_value(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// A described weight (an aligned stand-in address, never read), as the selector goldens build one.
std::optional<test::quantized_weight::PackedWeight> packed(QType qtype, int n, int k) {
    try {
        return test::quantized_weight::make_patterned_weight(qtype, n, k, 7U);
    } catch (const std::invalid_argument&) { return std::nullopt; }
}

// The kernel-facts decoder of a min-term format (the snapshot key's spelling).
std::string decoder_of(QType q) {
    switch (q) {
    case QType::GGML_Q4_K:
        return "GgmlQ45KQuadDecoder<4>";
    case QType::GGML_Q5_K:
        return "GgmlQ45KQuadDecoder<5>";
    case QType::GGML_Q2_K:
        return "GgmlQ2KDecoder";
    case QType::GGML_Q4_1:
        return "Ggml32Decoder<1>";
    case QType::GGML_Q5_1:
        return "Ggml32Decoder<3>";
    default:
        return "";
    }
}

// Whether the snapshot rows of `arch` admit the plane for the decoder and epilogue, the rule of
// rmma_plane_admitted (generic_gemm_launch.cuh) derived here from the rows themselves: every
// schedule's plane instance, plain and latency form, beside its twin with the MMA with ones (both
// rows or neither), with no more static shared memory, no more spill and at least as many CTAs
// per SM by registers and threads on the current (substituted) board.
bool admitted_by_rows(const std::string& arch, const std::string& decoder, int epilogue) {
    bool any = false;
    for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
        const auto s = generic::rmma_schedule_info(i);
        for (const bool latency : {false, true}) {
            kf::RmmaKernelKey key{decoder,   s.warps_m, s.warps_k, s.m_tiles, s.n_tiles,
                                  s.cluster, epilogue,  latency,   2};
            const auto plane = kf::rmma_facts(arch, key);
            key.sums         = 0;
            const auto ones  = kf::rmma_facts(arch, key);
            if (!plane && !ones) continue;
            if (!plane || !ones) return false;
            any               = true;
            const int threads = 32 * s.warps_m * s.warps_k;
            const auto ctas   = [&](const kf::KernelFacts& f) {
                return resident_ctas_per_sm(device_class(),
                                              KernelShape{f.registers, 0, threads, 0});
            };
            if (plane->static_shared_bytes > ones->static_shared_bytes ||
                !generic::rmma_plane_rule(plane->spill_bytes(), ones->spill_bytes(), ctas(*plane),
                                          ctas(*ones))) {
                return false;
            }
        }
    }
    return any;
}

// Whether this build's binary holds a plane instance of the decoder: its own snapshot's rows
// (kernel_facts::build_arch, the rows ninferno_kernel_facts_test checks against the binary). A
// substituted board's rows say what its release engine would launch; this binary can launch the
// plane only where it compiled one (the sm_70 build has none: kRmmaPlaneVariant), so a board
// substituted on it never reads the plane, whatever its own rows admit.
bool plane_in_build(const std::string& decoder) {
    const std::string build(kf::build_arch());
    for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
        const auto s = generic::rmma_schedule_info(i);
        for (const bool latency : {false, true}) {
            for (const int e : {0, 1, 2}) {
                const kf::RmmaKernelKey key{decoder,   s.warps_m, s.warps_k, s.m_tiles, s.n_tiles,
                                            s.cluster, e,         latency,   2};
                if (kf::rmma_facts(build, key)) return true;
            }
        }
    }
    return false;
}

void host_rules(const std::string& board) {
    using C = WeightCodec;
    check(generic::rmma_group_sums_values(C::GgmlQ4K) == 32, "Q4_K groups of 32");
    check(generic::rmma_group_sums_values(C::GgmlQ5K) == 32, "Q5_K groups of 32");
    check(generic::rmma_group_sums_values(C::GgmlQ41) == 32, "Q4_1 groups of 32");
    check(generic::rmma_group_sums_values(C::GgmlQ51) == 32, "Q5_1 groups of 32");
    check(generic::rmma_group_sums_values(C::GgmlQ2K) == 16, "Q2_K groups of 16");
    for (const C c : {C::GgmlQ6K, C::GgmlQ3K, C::GgmlQ40, C::GgmlQ50, C::Bf16, C::Q4}) {
        check(generic::rmma_group_sums_values(c) == 0, "no min term, no plane");
        check(generic::rmma_group_sums_bytes(c, 4096, 64) == 0, "no plane, no workspace");
        check(!generic::rmma_group_sums(c), "no min term: never the plane");
    }
    // Rows of whole 128-value steps: K / G rounded up to 128 / G.
    check(generic::rmma_group_sums_ld(4096, 32) == 128, "ld 4096 / 32");
    check(generic::rmma_group_sums_ld(4096, 16) == 256, "ld 4096 / 16");
    check(generic::rmma_group_sums_ld(1120, 32) == 36, "ld of a K ending inside a step");
    check(generic::rmma_group_sums_ld(96, 32) == 4, "ld of one partial step");
    const std::size_t bytes = generic::rmma_group_sums_bytes(C::GgmlQ4K, 4096, 32);
    check(bytes >= 32U * 128U * 4U && bytes < 32U * 128U * 4U + 256U, "plane bytes, aligned");

    // The profiles (the device's tensor-core MMA: no plane on Volta).
    const bool tensor = [] {
        try {
            return device_class().features.mma_tensor;
        } catch (const std::exception&) { return true; }
    }();
    for (const C c : {C::GgmlQ4K, C::GgmlQ5K, C::GgmlQ2K, C::GgmlQ41, C::GgmlQ51}) {
        {
            const ops::ScopedPrefillRoutes speed(false);
            check(!generic::rmma_group_sums(c),
                  "the speed profile keeps the MMA with ones and the adds (the price's)");
        }
        {
            const ops::ScopedPrefillRoutes prefill(true);
            check(generic::rmma_group_sums(c) == tensor,
                  "the prefill profile reads the plane where the board has tensor-core MMA");
            const ops::ScopedInvariantRoutes invariant;
            check(!generic::rmma_group_sums(c), "the invariant profile never reads the plane");
        }
    }

    // route_query answers the profile it is asked, whatever profile its caller is in (codex on
    // #709: the plane's resources differ from the MMA with ones', so an ambient prefill scope could
    // move a speed query's route).
    try {
        const DeviceClass d = device_class();
        for (const QType q : {QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q4_1}) {
            for (const std::int32_t t : {17, 64, 128}) {
                for (const bool prefill : {false, true}) {
                    const auto plain =
                        generic::route_query(d, q, ops::LinearPolicy::A16Only,
                                             generic::Epilogue::Store, 4096, 4096, t, prefill);
                    const ops::ScopedPrefillRoutes ambient(!prefill);
                    const auto inside =
                        generic::route_query(d, q, ops::LinearPolicy::A16Only,
                                             generic::Epilogue::Store, 4096, 4096, t, prefill);
                    check(plain == inside,
                          board + ": route_query takes the asked profile, not the caller's");
                }
            }
        }
    } catch (const std::exception&) {
        // Without a device the MMQ gate throws: nothing to compare there.
    }

    // rmma_plane_taken and resolve_route against the snapshot rows: one answer per format and
    // epilogue for every schedule and width, and only where this build compiled the plane; the
    // plane where the call's workspace holds it, else the MMA with ones said as a demotion; the
    // plan's capacity holds it.
    const std::string arch = kf::sass_arch().value_or("");
    check(kf::snapshot_header(kf::build_arch()).has_value(),
          "this build's architecture has a kernel-facts snapshot");
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    std::string admitted_list;
    for (const QType q : {QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q2_K, QType::GGML_Q4_1,
                          QType::GGML_Q5_1}) {
        const auto host = packed(q, 4096, 4096);
        if (!host) {
            check(false, "the test packer builds the min-term formats");
            continue;
        }
        const Weight w      = host->device_weight(payload);
        const WeightCodec c = generic::codec_of(w);
        for (const generic::Epilogue e : {generic::Epilogue::Store, generic::Epilogue::AddResidual,
                                          generic::Epilogue::SwiGlu}) {
            const bool expected = tensor && !arch.empty() && plane_in_build(decoder_of(q)) &&
                                  admitted_by_rows(arch, decoder_of(q), static_cast<int>(e));
            const std::string at = board + " " + std::string(generic::qtype_name(q)) + " " +
                                   std::to_string(static_cast<int>(e));
            if (expected) {
                admitted_list += " " + std::string(generic::qtype_name(q)) +
                                 (e == generic::Epilogue::Store         ? "/store"
                                  : e == generic::Epilogue::AddResidual ? "/add"
                                                                        : "/swiglu");
            }
            const std::int32_t rows = e == generic::Epilogue::SwiGlu ? 2048 : 4096;
            for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
                const auto s = generic::rmma_schedule_info(i);
                if (!generic::rmma_schedule_launchable(i) ||
                    !generic::rmma_tile_compiled(c, s.m_tiles, s.n_tiles, 4096)) {
                    continue;
                }
                const generic::Route r{generic::Family::RegisterMma, static_cast<std::uint8_t>(i)};
                const std::string ri = at + " " + generic::route_name(r);
                const ops::ScopedPrefillRoutes prefill(true);
                for (const std::int32_t t : {5, 32, 512}) {
                    check(generic::rmma_plane_taken(i, w, e, t, rows) == expected,
                          ri + " T " + std::to_string(t) + ": the plane as the snapshot rows say");
                    const auto described = generic::rmma_resources(i, w, e, t, rows);
                    check(!described || described->sums != 2,
                          ri + ": the price describes the instance with the MMA with ones");
                }
                const std::size_t need = generic::rmma_group_sums_bytes(c, 4096, 32);
                const auto resolved =
                    generic::resolve_route(r, w,
                                           generic::CallLayout{.ldx            = 4096,
                                                               .tokens         = 32,
                                                               .rows           = rows,
                                                               .epilogue       = e,
                                                               .workspace_free = need});
                check(resolved.group_sums == expected && resolved.reason == generic::Demotion::None,
                      ri + ": resolve_route takes the plane where the facts admit it");
                const ops::ScopedPrefillRoutes speed(false);
                const auto in_speed =
                    generic::resolve_route(r, w,
                                           generic::CallLayout{.ldx            = 4096,
                                                               .tokens         = 32,
                                                               .rows           = rows,
                                                               .epilogue       = e,
                                                               .workspace_free = need});
                check(!in_speed.group_sums && in_speed.reason == generic::Demotion::None,
                      ri + ": the speed profile takes no plane and is not demoted");
            }
            if (e != generic::Epilogue::Store) continue;
            // The demotions, on schedule 0 (w1x4, every format's).
            const generic::Route r{generic::Family::RegisterMma, 0};
            const auto resolve = [&](generic::Route route, int tokens,
                                     std::optional<std::size_t> free) {
                return generic::resolve_route(
                    route, w,
                    generic::CallLayout{
                        .ldx = 4096, .tokens = tokens, .rows = 4096, .workspace_free = free});
            };
            const ops::ScopedPrefillRoutes prefill(true);
            const std::size_t need = generic::rmma_group_sums_bytes(c, 4096, 32);
            const auto short_      = resolve(r, 32, need - 1);
            check(!short_.group_sums &&
                      short_.reason ==
                          (expected ? generic::Demotion::GroupSums : generic::Demotion::None) &&
                      short_.executed == r,
                  at + ": a workspace short of the plane: the same route, the MMA with ones, "
                       "demoted where the plane was taken");
            const auto none = resolve(r, 32, std::nullopt);
            check(!none.group_sums && none.reason == (expected ? generic::Demotion::GroupSums
                                                               : generic::Demotion::None),
                  at + ": no workspace: the MMA with ones, demoted where the plane was taken");
            // A call demoted before keeps its first reason: R8 given more tokens than it fits
            // and no workspace runs the register MMA, said as R8's demotion.
            const generic::Route r8{generic::Family::R8Mma, 0};
            try {
                const auto from_r8 = resolve(r8, 512, std::nullopt);
                check(!(from_r8.executed.family == generic::Family::RegisterMma) ||
                          (from_r8.reason != generic::Demotion::GroupSums &&
                           from_r8.reason != generic::Demotion::None),
                      at + ": an earlier demotion's reason stands");
            } catch (const std::exception&) {} // a format R8 does not read: nothing to demote
            const generic::Problem problem{q, ops::LinearPolicy::A16Only, 4096, 4096,
                                           generic::Epilogue::Store};
            check(!tensor || generic::workspace_capacity_bytes(problem, &w, 32, 32) >= need,
                  at + ": the plan's workspace holds the plane");
            check(!tensor || generic::workspace_capacity_bytes(problem, &w, 1, 2048) >=
                                 generic::rmma_group_sums_bytes(c, 4096, 2048),
                  at + ": the plan's workspace holds the plane of the interval's widest call");
            // The pure query resolves in the profile it is asked for, whatever the caller's: a
            // prefill tail of 5 tokens reads the plane where the facts admit it, the same call in
            // the speed profile does not.
            const ops::ScopedPrefillRoutes speed(false);
            const std::size_t need5 = generic::rmma_group_sums_bytes(c, 4096, 5);
            const auto tail =
                generic::resolved_route_query(device_class(), q, ops::LinearPolicy::A16Only,
                                              generic::Epilogue::Store, 4096, 4096, 5, need5, true);
            const auto decode5 = generic::resolved_route_query(
                device_class(), q, ops::LinearPolicy::A16Only, generic::Epilogue::Store, 4096, 4096,
                5, need5, false);
            check(tail && (tail->executed.family != generic::Family::RegisterMma ||
                           tail->group_sums == expected),
                  at + ": a prefill query below 17 tokens resolves with the plane the facts admit");
            check(decode5 && !decode5->group_sums,
                  at + ": a speed-profile query of 5 tokens takes no plane");
        }
    }
    check(generic::demotion_name(generic::Demotion::GroupSums) == "group_sums",
          "the demotion's name");
    // The rule on its inputs.
    check(generic::rmma_plane_rule(8, 8, 2, 2) && generic::rmma_plane_rule(0, 8, 3, 2),
          "no more spill and as many CTAs: the plane");
    check(!generic::rmma_plane_rule(16, 8, 2, 2), "more spill: the twin");
    check(!generic::rmma_plane_rule(0, 0, 1, 2), "fewer CTAs per SM: the twin");
    check(!generic::rmma_plane_rule(0, 0, 0, 0), "unknown facts: the twin");
    std::cout << board << " (sm_" << arch << "): the plane admitted for"
              << (admitted_list.empty() ? " none" : admitted_list)
              << (plane_in_build(decoder_of(QType::GGML_Q4_K))
                      ? ""
                      : " (this sm_" + std::string(kf::build_arch()) + " build has no plane)")
              << '\n';
}

// The pre-pass on the GPU against the CPU's tree, bit for bit.
int prepass_case(int k, int tokens, int values, int spread, std::uint32_t seed, bool aligned) {
    const int ldx = aligned ? k : k + 1; // an odd stride: the 2-byte loads
    std::vector<std::uint16_t> x(static_cast<std::size_t>(tokens) * ldx + 1);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> mant(0, 127), sign(0, 1), expo(-spread / 2, spread / 2);
    for (auto& v : x) {
        const int e = 127 + (spread > 0 ? expo(rng) : 0);
        v           = static_cast<std::uint16_t>((sign(rng) << 15) | (e << 7) | mant(rng));
    }
    // A block of zeros and one of BF16's largest values (whose sum is +inf) in the first token.
    if (k >= 2 * values) {
        for (int i = 0; i < values; ++i) x[static_cast<std::size_t>(i)] = 0;
        for (int i = values; i < 2 * values; ++i) x[static_cast<std::size_t>(i)] = 0x7f7fU;
    }
    const std::size_t offset = aligned ? 0 : 1; // x off its 16-byte word
    const int ld             = generic::rmma_group_sums_ld(k, values);
    const int poison_rows    = 2;
    std::uint16_t* dx        = nullptr;
    float* dout              = nullptr;
    const std::size_t out_n  = static_cast<std::size_t>(tokens + poison_rows) * ld;
    if (cudaMalloc(&dx, x.size() * 2) != cudaSuccess ||
        cudaMalloc(&dout, out_n * sizeof(float)) != cudaSuccess) {
        std::cerr << "cudaMalloc failed\n";
        return 1;
    }
    cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(dout, 0xff, out_n * sizeof(float));
    generic::GemmArgs args;
    args.x                 = dx + offset;
    args.ldx               = ldx;
    args.k                 = k;
    args.tokens            = tokens;
    args.group_sums_ld     = ld;
    args.group_sums_values = values;
    generic::launch_rmma_group_sums(args, dout, nullptr);
    std::vector<float> got(out_n);
    const bool ok_run =
        cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(got.data(), dout, out_n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(dx);
    cudaFree(dout);
    const std::string label = "pre-pass K " + std::to_string(k) + " T " + std::to_string(tokens) +
                              " G " + std::to_string(values) + " spread " + std::to_string(spread) +
                              (aligned ? "" : " unaligned");
    if (!ok_run) {
        std::cerr << label << ": the launch failed\n";
        return 1;
    }
    int bad = 0;
    for (int t = 0; t < tokens + poison_rows; ++t) {
        for (int g = 0; g < ld; ++g) {
            const float v = got[static_cast<std::size_t>(t) * ld + g];
            std::uint32_t bits;
            std::memcpy(&bits, &v, 4);
            if (t >= tokens) {
                bad += bits != 0xffffffffU; // a padded token's row is never written
                continue;
            }
            float want = 0.0F;
            if (g * values < k) {
                std::vector<float> vals(static_cast<std::size_t>(values));
                for (int i = 0; i < values; ++i) {
                    vals[static_cast<std::size_t>(i)] =
                        bf16_value(x[offset + static_cast<std::size_t>(t) * ldx + g * values + i]);
                }
                want = tree_sum(vals.data(), values);
            }
            std::uint32_t want_bits;
            std::memcpy(&want_bits, &want, 4);
            if (bits != want_bits) {
                if (bad < 3) {
                    std::cerr << label << ": token " << t << " group " << g << " got " << v
                              << " want " << want << '\n';
                }
                ++bad;
            }
        }
    }
    if (bad != 0) std::cerr << label << ": " << bad << " entries differ\n";
    return bad != 0 ? 1 : 0;
}

// A prefill-profile call (T = 5, 24, 32) captured into a CUDA graph with its pre-pass and replayed
// twice gives the eager call's bytes; the plane's workspace scope gives the pre-pass the same
// address at every replay. Q4_K, Q2_K and Q4_1, a one-tile and a four-tile route; where the
// device's facts do not admit the plane the graph holds the kernel alone.
int graph_capture() {
    int bad = 0, with_plane = 0, captured = 0;
    for (const QType q : {QType::GGML_Q4_K, QType::GGML_Q2_K, QType::GGML_Q4_1}) {
        const auto host = packed(q, 512, 1024);
        if (!host) return bad + 1;
        void* dw = nullptr;
        if (cudaMalloc(&dw, host->payload.size()) != cudaSuccess) return bad + 1;
        cudaMemcpy(dw, host->payload.data(), host->payload.size(), cudaMemcpyHostToDevice);
        const Weight w = host->device_weight(dw);
        for (const int tokens : {5, 24, 32}) {
            std::vector<std::uint16_t> xh(static_cast<std::size_t>(1024) * tokens);
            std::mt19937 rng(static_cast<unsigned>(tokens) + 5U);
            std::uniform_int_distribution<int> mant(0, 127), expo(120, 128), sign(0, 1);
            for (auto& v : xh)
                v = static_cast<std::uint16_t>((sign(rng) << 15) | (expo(rng) << 7) | mant(rng));
            void *dx = nullptr, *dy = nullptr;
            const std::size_t ybytes = static_cast<std::size_t>(512) * tokens * 2;
            cudaMalloc(&dx, xh.size() * 2);
            cudaMalloc(&dy, ybytes);
            cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
            // A pageable H2D may return before its DMA lands, and the launches below run on a
            // non-blocking stream that does not wait for the legacy stream.
            cudaDeviceSynchronize();
            Tensor tx(dx, DType::BF16, {1024, tokens});
            Tensor ty(dy, DType::BF16, {512, tokens});
            DeviceArena workspace(std::max<std::size_t>(
                generic::rmma_group_sums_bytes(generic::codec_of(w), 1024, tokens), 256));
            generic::Call call;
            call.x               = &tx;
            call.weight          = &w;
            call.rows            = 512;
            call.out[0]          = &ty;
            call.segment_rows[0] = 512;
            const ops::ScopedPrefillRoutes prefill(true);
            int four = 0; // m1.n4.w4x1
            for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
                const auto si = generic::rmma_schedule_info(i);
                if (si.warps_m == 4 && si.warps_k == 1 && si.m_tiles == 1 && si.n_tiles == 4 &&
                    si.cluster == 1)
                    four = i;
            }
            for (const int schedule : {0, four}) {
                const generic::Route r{generic::Family::RegisterMma,
                                       static_cast<std::uint8_t>(schedule)};
                const std::string label = std::string(generic::qtype_name(q)) + " T " +
                                          std::to_string(tokens) + " " + generic::route_name(r);
                const bool takes =
                    generic::rmma_plane_taken(schedule, w, generic::Epilogue::Store, tokens, 512);
                with_plane += takes ? 1 : 0;
                ++captured;
                cudaStream_t s = nullptr;
                cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
                std::vector<std::uint16_t> eager(ybytes / 2), replay(ybytes / 2);
                cudaMemsetAsync(dy, 0, ybytes, s);
                generic::run_route(r, call, &workspace, s);
                cudaStreamSynchronize(s);
                cudaMemcpy(eager.data(), dy, ybytes, cudaMemcpyDeviceToHost);
                cudaGraph_t graph    = nullptr;
                cudaGraphExec_t exec = nullptr;
                bool ok =
                    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
                if (ok) {
                    generic::run_route(r, call, &workspace, s);
                    ok = cudaStreamEndCapture(s, &graph) == cudaSuccess &&
                         cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
                }
                std::size_t nodes = 0;
                if (ok) cudaGraphGetNodes(graph, nullptr, &nodes);
                for (int rep = 0; ok && rep < 2; ++rep) {
                    cudaMemsetAsync(dy, 0xff, ybytes, s);
                    ok = cudaGraphLaunch(exec, s) == cudaSuccess &&
                         cudaStreamSynchronize(s) == cudaSuccess;
                    cudaMemcpy(replay.data(), dy, ybytes, cudaMemcpyDeviceToHost);
                    if (ok && replay != eager) {
                        std::cerr << label << ": replay " << rep
                                  << " differs from the eager call\n";
                        ++bad;
                    }
                }
                if (!ok) {
                    std::cerr << label << ": capture or replay failed\n";
                    ++bad;
                }
                // The pre-pass and the kernel: two kernel nodes (the kernel alone without the
                // plane).
                check(!ok || nodes == (takes ? 2U : 1U),
                      label + ": the graph holds the pre-pass where the call reads the plane, and "
                              "the kernel");
                if (exec != nullptr) cudaGraphExecDestroy(exec);
                if (graph != nullptr) cudaGraphDestroy(graph);
                cudaStreamDestroy(s);
            }
            cudaFree(dx);
            cudaFree(dy);
        }
        cudaFree(dw);
    }
    if (bad == 0)
        std::cout << "graph capture: " << captured << " prefill calls (" << with_plane
                  << " with the pre-pass and the plane's kernel) replay as run eagerly\n";
    return bad;
}

int gpu_checks() {
    int bad            = 0;
    std::uint32_t seed = 1;
    for (const int values : {32, 16}) {
        for (const int k : {256, 1024, 3840, 4096, 12288, 15360}) {
            for (const int tokens : {1, 7, 17, 24, 33, 300}) {
                for (const int spread : {0, 8, 16, 24}) {
                    bad += prepass_case(k, tokens, values, spread, ++seed, true);
                }
            }
            bad += prepass_case(k, 33, values, 16, ++seed, false);
        }
    }
    // Q4_1 / Q5_1: K of whole 32-value blocks, not of 128-value steps.
    for (const int k : {96, 4000, 1120}) {
        for (const int tokens : {1, 17, 300}) {
            bad += prepass_case(k, tokens, 32, 16, ++seed, true);
            bad += prepass_case(k, tokens, 32, 24, ++seed, false);
        }
    }
    {
        generic::GemmArgs args;
        args.k = 100, args.tokens = 1, args.group_sums_values = 32, args.group_sums_ld = 4;
        bool threw = false;
        try {
            generic::launch_rmma_group_sums(args, nullptr, nullptr);
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "the pre-pass refuses a K of no whole groups");
    }
    if (bad == 0) std::cout << "pre-pass: every case equals the CPU's tree bit for bit\n";

    // This GPU's own rules (its build's snapshot and its SM limits).
    host_rules("this GPU");

    // The instance the price describes: the MMA with ones' (key sums 0) or, in the speed profile
    // at decode widths in a tile of four or more token tiles, the adds' (1); never the plane's (2),
    // which has no price yet. The plane's instance a prefill call takes agrees with the binary.
    const DeviceFeatures features = device_class().features;
    if (!features.mma_tensor) return bad;           // Volta: its own kernels, no plane (host_rules)
    const bool adds     = features.mma_bf16_native; // rmma_add_sums: not Turing's f16 MMA
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    for (const QType q : {QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q2_K, QType::GGML_Q4_1,
                          QType::GGML_Q5_1}) {
        const auto host = packed(q, 4096, 4096);
        if (!host) return bad + 1;
        const Weight w = host->device_weight(payload);
        for (const generic::Epilogue e : {generic::Epilogue::Store, generic::Epilogue::AddResidual,
                                          generic::Epilogue::SwiGlu}) {
            const std::int32_t rows = e == generic::Epilogue::SwiGlu ? 2048 : 4096;
            const auto described    = [&](int schedule, int tokens) -> std::optional<int> {
                kf::LiveCheckSink reads;
                {
                    const kf::ScopedLiveCheck live(reads);
                    (void)generic::rmma_resources(schedule, w, e, tokens, rows);
                }
                if (reads.empty()) return std::nullopt;
                return reads.back().key.sums;
            };
            for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
                if (!generic::rmma_schedule_launchable(i)) continue;
                const auto s = generic::rmma_schedule_info(i);
                if (!generic::rmma_tile_compiled(generic::codec_of(w), s.m_tiles, s.n_tiles, 4096))
                    continue;
                const std::string at = " (" + std::string(generic::qtype_name(q)) + " " +
                                       std::to_string(static_cast<int>(e)) + " " +
                                       generic::route_name({generic::Family::RegisterMma,
                                                            static_cast<std::uint8_t>(i)}) +
                                       ")";
                {
                    const ops::ScopedPrefillRoutes speed(false);
                    const int twin  = adds && s.n_tiles >= 4 ? 1 : 0;
                    const auto wide = described(i, 32);
                    check(wide && *wide == twin,
                          "T = 32 describes the MMA with ones or the adds" + at);
                    const auto narrow = described(i, 16);
                    check(narrow && *narrow == twin,
                          "T = 16 describes the MMA with ones or the adds" + at);
                }
                {
                    const ops::ScopedPrefillRoutes prefill(true);
                    const auto tail = described(i, 5);
                    check(tail && *tail == 0, "a prefill call describes the MMA with ones" + at);
                    kf::LiveCheckSink reads;
                    bool takes = false;
                    {
                        const kf::ScopedLiveCheck live(reads);
                        takes = generic::rmma_plane_taken(i, w, e, 5, rows);
                    }
                    bool read_plane = false;
                    for (const kf::LiveRead& read : reads) {
                        if (read.key.sums != 2) continue;
                        read_plane = true;
                        check(read.agrees(), "the plane's instance agrees with the binary" + at +
                                                 ": " + kf::describe(read));
                    }
                    check(read_plane == takes, "the plane's instance is the one described" + at);
                    const ops::ScopedInvariantRoutes invariant;
                    const auto exact = described(i, 32);
                    check(exact && *exact == 0,
                          "the invariant profile describes the MMA with ones" + at);
                }
            }
        }
    }
    bad += graph_capture();
    return bad;
}

} // namespace

int main() {
    // The rules on substituted boards (no GPU asked): Blackwell, Ampere, Turing and Volta.
    int boards = 0;
    for (const generic::FactsBoard& b : generic::kFactsBoards) {
        const std::string name = b.name;
        if (name != "rtx5090" && name != "rtx3090" && name != "rtx2080ti" && name != "v100") {
            continue;
        }
        const ScopedDeviceClass scoped(generic::board_device_class(b));
        host_rules(name);
        ++boards;
    }
    check(boards == 4, "the four boards of the host rules are in the facts table");
    int count      = 0;
    const bool gpu = cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    if (gpu) failures += gpu_checks();
    if (failures != 0) {
        std::cerr << failures << " group-sums check(s) failed\n";
        return 1;
    }
    std::cout << "group sums: host rules hold" << (gpu ? "; GPU checks pass" : "; no GPU, skipped")
              << '\n';
    return gpu ? 0 : 77;
}
