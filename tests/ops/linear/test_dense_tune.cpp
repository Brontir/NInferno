// The generic path's dense tuner (ops/generic/generic_tune.cpp): a problem's first eager call times
// it, the tuned route replaces the heuristic's only within its activation class (A16 for A16, an R8
// schedule for R8, never one that needs another workspace), an R8 call keeps its bytes, a prefill
// band's route gives a token the heuristic's bytes (a W4A8 schedule for W4A8, the prefill route's
// summation order for A16), the cache file gives a second process the same routes without timing
// (and times again only the bands a failure skipped), and a call inside a CUDA graph capture times
// nothing.
#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "ninferno/ops/route_table.h"
#include "ops/generic/generic_gemm.h"
#include "support/temp_dir.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

// The tuner's file under the test's --tune-cache directory (set_dense_tune).
std::filesystem::path g_cache_file;

using namespace ninferno;
using namespace ninferno::test;
using namespace ninferno::test::linear;
namespace generic = ninferno::ops::detail::generic;

struct Case {
    const char* name;
    QType qtype;
    ops::LinearPolicy policy;
    std::int32_t n, k;
};

struct Device {
    quantized_weight::PackedWeight host;
    DeviceBuffer payload;
    Weight weight;
};

Device upload(const Case& c, std::uint32_t seed) {
    if (c.qtype == QType::BF16) {
        // A plain BF16 matrix (the packed fixtures are quantized formats).
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0F, 0.05F);
        std::vector<__nv_bfloat16> host(static_cast<std::size_t>(c.n) * c.k);
        for (auto& v : host) v = __float2bfloat16(normal(rng));
        Device d{};
        d.payload = DeviceBuffer(host.size() * 2);
        d.payload.copy_from_host(host.data(), d.payload.bytes);
        d.weight.qtype         = QType::BF16;
        d.weight.layout        = QuantLayout::Contiguous;
        d.weight.qdata         = d.payload.p;
        d.weight.payload       = d.payload.p;
        d.weight.payload_bytes = d.payload.bytes;
        d.weight.n = d.weight.shape[0] = d.weight.padded_shape[0] = c.n;
        d.weight.k = d.weight.shape[1] = d.weight.padded_shape[1] = c.k;
        d.weight.ndim                                             = 2;
        return d;
    }
    Device d{quantized_weight::make_patterned_weight(c.qtype, c.n, c.k, seed), {}, {}};
    d.payload = DeviceBuffer(d.host.payload.size());
    d.payload.copy_from_host(d.host.payload.data(), d.payload.bytes);
    d.weight = d.host.device_weight(d.payload.p);
    return d;
}

std::vector<std::uint16_t> download(const void* p, std::size_t count) {
    std::vector<std::uint16_t> out(count);
    cuda_check(cudaMemcpy(out.data(), p, count * 2, cudaMemcpyDeviceToHost), "download");
    return out;
}

// One store call of `tokens` columns on `route` (nullopt: the dispatcher's own choice).
std::vector<std::uint16_t> call(const Device& d, const DeviceBuffer& x, std::int32_t tokens,
                                std::optional<generic::Route> route, ops::LinearPolicy policy,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    DeviceBuffer out(static_cast<std::size_t>(d.weight.n) * tokens * 2);
    Tensor tx(x.p, DType::BF16, {d.weight.k, tokens});
    Tensor tout(out.p, DType::BF16, {d.weight.n, tokens});
    workspace.reset();
    if (route) {
        generic::Call c;
        c.x               = &tx;
        c.weight          = &d.weight;
        c.rows            = d.weight.n;
        c.out[0]          = &tout;
        c.segment_rows[0] = d.weight.n;
        generic::run_route(*route, c, &workspace, stream);
    } else {
        generic::linear(tx, d.weight, tout, policy, &workspace, stream);
    }
    cuda_check(cudaStreamSynchronize(stream), "call");
    return download(out.p, static_cast<std::size_t>(d.weight.n) * tokens);
}

int check_case(const Case& c, cudaStream_t stream) {
    int failures = 0;
    const Device d = upload(c, 17U + static_cast<std::uint32_t>(c.n));
    const std::int32_t wide = 128;
    const auto activation = make_linear_activation(c.k, wide, 5U, ActivationCompute::A16);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    DeviceArena workspace(std::max<std::size_t>(
        generic::route_workspace_bytes({generic::Family::W4a8Mma, 0}, c.k, wide), 256));
    const generic::Problem problem{c.qtype, c.policy, c.n, c.k, generic::Epilogue::Store};

    // Nothing is tuned before the first eager call; the call tunes it.
    for (std::int32_t t = 1; t <= generic::kDenseTuneMaxTokens; ++t) {
        if (generic::tuned_dense_route(problem, t)) {
            std::cerr << c.name << ": tuned before any call at T=" << t << "\n";
            ++failures;
        }
    }
    (void)call(d, x, wide, std::nullopt, c.policy, workspace, stream);
    const auto report = generic::dense_tune_report();
    if (report.empty() || report.back().find("dense tune") == std::string::npos) {
        std::cerr << c.name << ": the first eager call timed nothing\n";
        return failures + 1;
    }
    std::cout << report.back() << "\n";

    std::vector<std::optional<generic::Route>> tuned(generic::kDenseTuneMaxTokens + 1);
    for (std::int32_t t = 1; t <= generic::kDenseTuneMaxTokens; ++t) {
        tuned[t] = generic::tuned_dense_route(problem, t);
        const auto heuristic = generic::heuristic_route(problem, d.weight, t, false);
        const auto chosen    = generic::select_route(problem, d.weight, t, false);
        if (!tuned[t]) {
            if (!(chosen == heuristic)) {
                std::cerr << c.name << " T=" << t << ": untuned width left the heuristic\n";
                ++failures;
            }
            continue;
        }
        if (!(chosen == *tuned[t])) {
            std::cerr << c.name << " T=" << t << ": dispatch ignores the tuned route\n";
            ++failures;
        }
        const bool r8 = heuristic.family == generic::Family::R8Mma;
        // Any admitted route may replace the vector kernel's rule (heuristic_route).
        if (heuristic.family == generic::Family::GgmlVec) continue;
        // ggml's matrix kernel may replace an A16 schedule at verify widths where A8 is allowed
        // (its workspace is the problem's capacity, workspace_capacity_bytes).
        if (tuned[t]->family == generic::Family::GgmlMmq) {
            if (r8 || c.policy == ops::LinearPolicy::A16Only || t < generic::kMmqMinTokens) {
                std::cerr << c.name << " T=" << t << ": ggml_mmq against the heuristic "
                          << generic::route_name(heuristic) << "\n";
                ++failures;
            }
            continue;
        }
        // R8 may replace an A16 schedule at a verify width (r8_verify_tunable; its workspace is the
        // problem's capacity).
        if (tuned[t]->family == generic::Family::R8Mma && !r8) {
            if (c.policy == ops::LinearPolicy::A16Only || t < 2 || t > generic::kR8VerifyTokens) {
                std::cerr << c.name << " T=" << t << ": R8 against the heuristic "
                          << generic::route_name(heuristic) << "\n";
                ++failures;
            }
            continue;
        }
        if ((tuned[t]->family == generic::Family::R8Mma) != r8 ||
            generic::route_workspace_bytes(*tuned[t], c.k, t) !=
                generic::route_workspace_bytes(heuristic, c.k, t)) {
            std::cerr << c.name << " T=" << t << ": tuned " << generic::route_name(*tuned[t])
                      << " against the heuristic " << generic::route_name(heuristic)
                      << " crosses the activation class or needs a workspace\n";
            ++failures;
        }
        // An R8 call keeps its bytes (decode and verify widths).
        if (r8 && call(d, x, t, heuristic, c.policy, workspace, stream) !=
                      call(d, x, t, std::nullopt, c.policy, workspace, stream)) {
            std::cerr << c.name << " T=" << t << ": the tuned R8 schedule changed the bytes\n";
            ++failures;
        }
    }

    // Every route a prefill band may take gives a token the heuristic's bytes, whatever the
    // timing picked: at a band's own width (17), inside one (47, 100) and at the wide tile (128).
    const std::vector<std::int32_t> widths = {17, 47, 100, 128};
    for (const std::int32_t t : widths) {
        const auto heuristic  = generic::heuristic_route(problem, d.weight, t, true);
        const auto bytes      = call(d, x, t, heuristic, c.policy, workspace, stream);
        const auto candidates = generic::prefill_band_candidates(heuristic, d.weight, c.policy);
        // Volta's tiled kernel takes the width whatever the bands (volta_tiled_route), as ggml's
        // matrix kernel does at prefill where its rule holds (mmq_prefill_wins): one route each.
        const bool volta = generic::volta_tiled_route(d.weight, c.policy, t).has_value();
        const bool mmq   = heuristic.family == generic::Family::GgmlMmq;
        if (candidates.size() < 2 && !volta && !mmq) {
            std::cerr << c.name << " T=" << t << ": no band candidate beside "
                      << generic::route_name(heuristic) << " to compare\n";
            ++failures;
        }
        for (const auto r : candidates) {
            if (call(d, x, t, r, c.policy, workspace, stream) != bytes) {
                std::cerr << c.name << " T=" << t << ": band candidate " << generic::route_name(r)
                          << " changed the bytes of " << generic::route_name(heuristic) << "\n";
                ++failures;
            }
        }
    }
    // Prefill bands (T >= 17) within the widths this test's activation holds.
    std::vector<std::optional<generic::Route>> bands(wide + 1);
    for (const std::int32_t t : widths) {
        const auto heuristic = generic::heuristic_route(problem, d.weight, t, true);
        bands[t]             = generic::tuned_prefill_route(problem, t, heuristic);
        const auto chosen    = generic::select_route(problem, d.weight, t, true);
        if (const auto volta = generic::volta_tiled_route(d.weight, c.policy, t)) {
            // The tiled kernel takes the width before any band, so the tuner times none there.
            if (bands[t] || !(chosen == *volta)) {
                std::cerr << c.name << " T=" << t << ": Volta's tiled route did not take the width"
                          << (bands[t] ? " (a band was timed)" : "") << "\n";
                ++failures;
            }
            continue;
        }
        if (!bands[t]) {
            if (!(chosen == heuristic)) {
                std::cerr << c.name << " T=" << t
                          << ": an untuned prefill band left the heuristic\n";
                ++failures;
            }
            continue;
        }
        std::cout << c.name << " T=" << t << ": prefill band " << generic::route_name(*bands[t])
                  << " (heuristic " << generic::route_name(heuristic) << ")\n";
        const bool w4a8 = heuristic.family == generic::Family::W4a8Mma;
        if (w4a8 ? bands[t]->family != generic::Family::W4a8Mma
                 : !generic::same_summation(*bands[t], heuristic)) {
            std::cerr << c.name << " T=" << t << ": the prefill band sums otherwise\n";
            ++failures;
        }
        if (!(chosen == *bands[t])) {
            std::cerr << c.name << " T=" << t << ": dispatch ignores the prefill band\n";
            ++failures;
        }
        if (call(d, x, t, heuristic, c.policy, workspace, stream) !=
            call(d, x, t, *bands[t], c.policy, workspace, stream)) {
            std::cerr << c.name << " T=" << t << ": the prefill band changed the bytes\n";
            ++failures;
        }
    }

    // A second process: the file gives the same routes without timing.
    generic::clear_dense_tune();
    generic::enable_dense_tune(true);
    (void)call(d, x, wide, std::nullopt, c.policy, workspace, stream);
    if (!generic::dense_tune_report().empty()) {
        std::cerr << c.name << ": the cached problem was timed again\n";
        ++failures;
    }
    for (std::int32_t t = 1; t <= generic::kDenseTuneMaxTokens; ++t) {
        const auto again = generic::tuned_dense_route(problem, t);
        if (again.has_value() != tuned[t].has_value() || (again && !(*again == *tuned[t]))) {
            std::cerr << c.name << " T=" << t << ": the cache gave another route\n";
            ++failures;
        }
    }
    for (const std::int32_t t : widths) {
        const auto again = generic::tuned_prefill_route(
            problem, t, generic::heuristic_route(problem, d.weight, t, true));
        if (again.has_value() != bands[t].has_value() || (again && !(*again == *bands[t]))) {
            std::cerr << c.name << " T=" << t << ": the cache gave another prefill band\n";
            ++failures;
        }
    }

    // Bands a failure skipped are timed again by the next process, alone: the decode widths stay.
    {
        const std::filesystem::path& path = g_cache_file;
        std::stringstream text;
        text << std::ifstream(path).rdbuf();
        std::string json        = text.str();
        const std::string field = "\"prefill_base\": {";
        const std::string mark  = "\"prefill_skipped\": \"test\", ";
        for (auto at = json.find(field); at != std::string::npos; at = json.find(field, at)) {
            json.insert(at, mark);
            at += mark.size() + field.size();
        }
        std::ofstream(path) << json;
    }
    generic::clear_dense_tune();
    generic::enable_dense_tune(true);
    (void)call(d, x, wide, std::nullopt, c.policy, workspace, stream);
    const auto retimed = generic::dense_tune_report();
    if (retimed.size() != 1 || retimed[0].find("(prefill bands)") == std::string::npos) {
        std::cerr << c.name << ": skipped bands were not timed again on their own\n";
        ++failures;
    }
    for (std::int32_t t = 1; t <= generic::kDenseTuneMaxTokens; ++t) {
        const auto again = generic::tuned_dense_route(problem, t);
        if (again.has_value() != tuned[t].has_value() || (again && !(*again == *tuned[t]))) {
            std::cerr << c.name << " T=" << t << ": retiming the bands changed a decode route\n";
            ++failures;
        }
    }
    generic::clear_dense_tune();
    generic::enable_dense_tune(true);
    (void)call(d, x, wide, std::nullopt, c.policy, workspace, stream);
    if (!generic::dense_tune_report().empty()) {
        std::cerr << c.name << ": the retimed bands were not saved\n";
        ++failures;
    }
    return failures;
}

// A problem first met inside a capture is not timed there (and allocates nothing).
int check_capture(cudaStream_t stream) {
    const Case c{"q8 capture", QType::Q8_G32_FP16, ops::LinearPolicy::A16Only, 1536, 2048};
    const Device d = upload(c, 99U);
    const auto activation = make_linear_activation(c.k, 4, 7U, ActivationCompute::A16);
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    DeviceBuffer out(static_cast<std::size_t>(c.n) * 4 * 2);
    Tensor tx(x.p, DType::BF16, {c.k, 4});
    Tensor tout(out.p, DType::BF16, {c.n, 4});
    const std::size_t before = generic::dense_tune_report().size();
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    generic::linear(tx, d.weight, tout, c.policy, nullptr, stream);
    cudaGraph_t graph = nullptr;
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda_check(cudaGraphDestroy(graph), "destroy graph");
    if (generic::dense_tune_report().size() != before) {
        std::cerr << "capture: a problem was timed inside a capture\n";
        return 1;
    }
    // Its next eager call times it.
    generic::linear(tx, d.weight, tout, c.policy, nullptr, stream);
    cuda_check(cudaStreamSynchronize(stream), "eager");
    if (generic::dense_tune_report().size() != before + 1) {
        std::cerr << "capture: the eager call after the capture timed nothing\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "CUDA unavailable; skipping dense tune test\n";
        return 77;
    }
    const TempDir scratch("ninferno_dense_tune_test");
    const auto& cache = scratch.path();
    g_cache_file = cache / "dense_tune.json";
    generic::set_dense_tune(ninferno::TuneMode::On, cache); // off by default
    // The heuristic is what the tuner is compared against: no class table between them.
    ops::clear_route_table();
    generic::enable_dense_tune(true);
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "stream");
    int failures = 0;
    // Q4_K under AllowA8: R8 at T = 1 where the device takes it, the register MMA above. Q8 G32
    // under A16Only: the register MMA at every decode width. The other W4A8 formats (their bands
    // reach schedules the heuristic never picks) and BF16 (staged and register MMA in one order).
    const Case cases[] = {
        {"q4k a8", QType::GGML_Q4_K, ops::LinearPolicy::AllowA8, 2048, 4096},
        {"q8 a16", QType::Q8_G32_FP16, ops::LinearPolicy::A16Only, 3072, 2048},
        {"q2k a8", QType::GGML_Q2_K, ops::LinearPolicy::AllowA8, 1536, 2048},
        {"iq3xxs a8", QType::GGML_IQ3_XXS, ops::LinearPolicy::AllowA8, 1536, 2048},
        {"q6k a8", QType::GGML_Q6_K, ops::LinearPolicy::AllowA8, 1536, 2048},
        {"bf16 a16", QType::BF16, ops::LinearPolicy::A16Only, 1536, 2048},
    };
    for (const Case& c : cases) failures += check_case(c, stream);
    failures += check_capture(stream);
    cuda_check(cudaStreamDestroy(stream), "stream");
    generic::enable_dense_tune(false);
    generic::set_dense_tune(ninferno::TuneMode::Off, {});
    if (failures == 0) std::cout << "dense tune: ok\n";
    return failures == 0 ? 0 : 1;
}
