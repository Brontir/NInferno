// The register MMA's pre-summed activation group plane (#619, GemmArgs::group_sums). Host: which
// calls take it (rmma_group_sums: the min-term formats on tensor-core MMA, every width of the
// prefill profile, 17 tokens or more in the speed profile, never in the invariant profile), its
// row length and workspace, the workspace a plan reserves for it, and resolve_route's demotion of
// a call without room for it. GPU (skipped without one): the pre-pass equals the CPU's tree sum
// bit for bit for every K, T, group size, exponent spread and alignment, writes zero past K and
// nothing for tokens past the call's; and the launch the price describes is the plane's instance
// where the call takes the plane and the MMA with ones' elsewhere (the kernel-facts key's `sums`).
#include "ops/quantized_weight.h"

#include "core/arena.h"
#include "core/device_class.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/kernel_facts.h"
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

void host_rules() {
    using C = WeightCodec;
    check(generic::rmma_group_sums_values(C::GgmlQ4K) == 32, "Q4_K groups of 32");
    check(generic::rmma_group_sums_values(C::GgmlQ5K) == 32, "Q5_K groups of 32");
    check(generic::rmma_group_sums_values(C::GgmlQ41) == 32, "Q4_1 groups of 32");
    check(generic::rmma_group_sums_values(C::GgmlQ51) == 32, "Q5_1 groups of 32");
    check(generic::rmma_group_sums_values(C::GgmlQ2K) == 16, "Q2_K groups of 16");
    for (const C c : {C::GgmlQ6K, C::GgmlQ3K, C::GgmlQ40, C::GgmlQ50, C::Bf16, C::Q4}) {
        check(generic::rmma_group_sums_values(c) == 0, "no min term, no plane");
        check(generic::rmma_group_sums_bytes(c, 4096, 64) == 0, "no plane, no workspace");
        check(!generic::rmma_group_sums(c, 64), "no min term: never the plane");
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
            check(!generic::rmma_group_sums(c, 1) && !generic::rmma_group_sums(c, 16),
                  "the speed profile keeps the MMA with ones below 17 tokens");
            check(generic::rmma_group_sums(c, 17) == tensor &&
                      generic::rmma_group_sums(c, 2048) == tensor,
                  "the speed profile reads the plane from 17 tokens");
        }
        {
            const ops::ScopedPrefillRoutes prefill(true);
            check(generic::rmma_group_sums(c, 1) == tensor &&
                      generic::rmma_group_sums(c, 5) == tensor &&
                      generic::rmma_group_sums(c, 512) == tensor,
                  "the prefill profile reads the plane at every width");
            const ops::ScopedInvariantRoutes invariant;
            check(!generic::rmma_group_sums(c, 5) && !generic::rmma_group_sums(c, 512),
                  "the invariant profile never reads the plane");
        }
    }

    // resolve_route: the plane where the call's workspace holds it, else the MMA with ones said
    // as a demotion; the plan's capacity holds it.
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    for (const QType q : {QType::GGML_Q4_K, QType::GGML_Q2_K, QType::GGML_Q4_1}) {
        const auto host = packed(q, 4096, 4096);
        if (!host) {
            check(false, "the test packer builds the min-term formats");
            continue;
        }
        const Weight w      = host->device_weight(payload);
        const WeightCodec c = generic::codec_of(w);
        const generic::Route r{generic::Family::RegisterMma, 0};
        const auto resolve = [&](int tokens, std::optional<std::size_t> free) {
            return generic::resolve_route(
                r, w,
                generic::CallLayout{
                    .ldx = 4096, .tokens = tokens, .rows = 4096, .workspace_free = free});
        };
        const ops::ScopedPrefillRoutes speed(false);
        const std::size_t need = generic::rmma_group_sums_bytes(c, 4096, 32);
        const auto fits        = resolve(32, need);
        check(fits.group_sums == tensor && fits.reason == generic::Demotion::None &&
                  fits.executed == r,
              "a workspace that holds the plane takes it");
        if (tensor) {
            const auto short_ = resolve(32, need - 1);
            check(!short_.group_sums && short_.reason == generic::Demotion::GroupSums &&
                      short_.executed == r,
                  "a workspace short of the plane: the same route, demoted to the MMA with ones");
            const auto none = resolve(32, std::nullopt);
            check(!none.group_sums && none.reason == generic::Demotion::GroupSums,
                  "no workspace: the MMA with ones, demoted");
        }
        const auto decode = resolve(16, need);
        check(!decode.group_sums && decode.reason == generic::Demotion::None,
              "a decode width takes no plane and is not demoted");
        const generic::Problem problem{q, ops::LinearPolicy::A16Only, 4096, 4096,
                                       generic::Epilogue::Store};
        check(!tensor || generic::workspace_capacity_bytes(problem, &w, 32, 32) >= need,
              "the plan's workspace holds the plane");
        check(!tensor || generic::workspace_capacity_bytes(problem, &w, 1, 2048) >=
                             generic::rmma_group_sums_bytes(c, 4096, 2048),
              "the plan's workspace holds the plane of the interval's widest call");
        check(generic::demotion_name(generic::Demotion::GroupSums) == "group_sums",
              "the demotion's name");
    }
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

// A verify-width call (T = 24, 32: speed profile, the plane) captured into a CUDA graph with its
// pre-pass and replayed twice gives the eager call's bytes; the plane's workspace scope gives the
// pre-pass the same address at every replay. Q4_K and Q2_K, a one-tile and a four-tile route.
int graph_capture() {
    int bad = 0;
    for (const QType q : {QType::GGML_Q4_K, QType::GGML_Q2_K}) {
        const auto host = packed(q, 512, 1024);
        if (!host) return bad + 1;
        void* dw = nullptr;
        if (cudaMalloc(&dw, host->payload.size()) != cudaSuccess) return bad + 1;
        cudaMemcpy(dw, host->payload.data(), host->payload.size(), cudaMemcpyHostToDevice);
        const Weight w = host->device_weight(dw);
        for (const int tokens : {24, 32}) {
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
            const ops::ScopedPrefillRoutes speed(false);
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
                // The pre-pass and the kernel: two kernel nodes.
                check(!ok || nodes == 2, label + ": the graph holds the pre-pass and the kernel");
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
        std::cout << "graph capture: the pre-pass and the plane's kernel replay as run eagerly\n";
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

    // The instance the price describes: the plane's where the call takes it (key sums 2), the MMA
    // with ones' in the invariant profile and at decode widths (0), the adds' (1) only at decode
    // widths in a tile of four or more token tiles.
    const DeviceFeatures features = device_class().features;
    if (!features.mma_tensor) return bad;           // Volta: its own kernels, no plane (host_rules)
    const bool adds     = features.mma_bf16_native; // rmma_add_sums: not Turing's f16 MMA
    void* const payload = reinterpret_cast<void*>(std::uintptr_t{1} << 32);
    const auto host     = packed(QType::GGML_Q4_K, 4096, 4096);
    if (!host) return bad + 1;
    const Weight w       = host->device_weight(payload);
    const auto described = [&](int schedule, int tokens) -> std::optional<int> {
        kf::LiveCheckSink reads;
        {
            const kf::ScopedLiveCheck live(reads);
            (void)generic::rmma_resources(schedule, w, generic::Epilogue::Store, tokens);
        }
        if (reads.empty()) return std::nullopt;
        return reads.back().key.sums;
    };
    for (int i = 0; i < generic::rmma_schedule_count(); ++i) {
        if (!generic::rmma_schedule_launchable(i)) continue;
        const auto s = generic::rmma_schedule_info(i);
        if (!generic::rmma_tile_compiled(generic::codec_of(w), s.m_tiles, s.n_tiles, 4096))
            continue;
        const std::string at =
            " (" +
            generic::route_name({generic::Family::RegisterMma, static_cast<std::uint8_t>(i)}) + ")";
        {
            const ops::ScopedPrefillRoutes speed(false);
            const auto wide = described(i, 32);
            check(wide && *wide == 2, "T = 32 describes the plane's instance" + at);
            const auto narrow = described(i, 16);
            check(narrow && *narrow == (adds && s.n_tiles >= 4 ? 1 : 0),
                  "T = 16 describes the MMA with ones or the adds" + at);
        }
        {
            const ops::ScopedPrefillRoutes prefill(true);
            const auto tail = described(i, 5);
            check(tail && *tail == 2, "a prefill tail describes the plane's instance" + at);
            const ops::ScopedInvariantRoutes invariant;
            const auto exact = described(i, 32);
            check(exact && *exact == 0, "the invariant profile describes the MMA with ones" + at);
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
        host_rules();
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
