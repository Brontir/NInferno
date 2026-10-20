// Gemma 4 projections: a token's output must not depend on how many other tokens share the call.
// A prefill chunk's size is decided by where a turn was cut, not by the model, so any column of a
// T-token call has to equal, byte for byte, the same column of a longer call over the same leading
// tokens. Every registered 26B-A4B / 31B shape is run once at the widest T and again at every T
// where a route or tile switches, and the leading columns are compared bitwise; windows further
// into the widest call check that a column does not depend on its place in the call either. The
// A16 rows are the planes the --a16 and --a16-full profiles run.
#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/weight.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninferno;
using namespace ninferno::test;
using namespace ninferno::test::linear;

struct ShapeUnderTest {
    const char* name;
    WeightGenerator generator;
    std::int32_t n;
    std::int32_t k;
    ops::LinearPolicy policy;
};

// The prefill chunk sizes of a session run T in [17, chunk]; decode and verify (T <= 16) are
// separate routes by design. Every frontier of the registered route tables is probed from both
// sides, plus ragged values inside the tiles.
constexpr std::int32_t kTokens[] = {17,  31,  32,  33,  44,  63,  64,  65,  100, 128, 129,
                                    160, 161, 224, 225, 236, 256, 257, 300, 448, 449, 512,
                                    513, 640, 641, 896, 897, 1000, 1024, 1025, 1348, 2048};
constexpr std::int32_t kWidest = 2048;

// A later chunk's tokens are the leading columns of their own call but sit anywhere in a wider
// one, so a column must also not depend on where it lies inside the call: windows [offset,
// offset+T) of the widest call recomputed as T-token calls. The offsets are prefill chunks a
// session can start at (700 and 1400 for a 700-token chunk) and ragged tile positions.
struct Window {
    std::int32_t offset, tokens;
};

constexpr Window kWindows[] = {{700, 700}, {1400, 648}, {33, 100}, {60, 17}, {1000, 1048}};

// Activations whose partial sums round in FP32: normal values spread over thirteen octaves. The
// patterned activations of the accuracy tests, and plain normal ones, left the narrow A16 tail
// schedules (another K order) invisible; with these, those schedules fail the windows and the
// T = 31/32 cases of several 31B shapes.
std::vector<std::uint16_t> random_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_int_distribution<int> octave(-6, 6);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(k) * t);
    for (auto& value : out) {
        const float f      = std::ldexp(normal(generator), octave(generator));
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        const std::uint32_t round = 0x7FFFU + ((bits >> 16) & 1U);
        value                     = static_cast<std::uint16_t>((bits + round) >> 16);
    }
    return out;
}

std::vector<std::uint16_t> run(const ShapeUnderTest& shape, const Weight& weight,
                               const DeviceBuffer& x_all, std::int32_t t, std::int32_t offset = 0) {
    Tensor x(static_cast<std::uint16_t*>(x_all.p) + static_cast<std::size_t>(offset) * shape.k,
             DType::BF16, {shape.k, t});
    DeviceBuffer device_out(static_cast<std::size_t>(shape.n) * t * sizeof(std::uint16_t));
    Tensor out(device_out.p, DType::BF16, {shape.n, t});
    const std::size_t capacity =
        ops::linear_workspace_capacity_bytes(weight.qtype, shape.n, shape.k, shape.policy, t, t);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::linear(x, weight, out, shape.policy, workspace, nullptr);
    cuda_check(cudaDeviceSynchronize(), "linear");
    std::vector<std::uint16_t> host(static_cast<std::size_t>(shape.n) * t);
    cuda_check(cudaMemcpy(host.data(), device_out.p, host.size() * 2, cudaMemcpyDeviceToHost),
               "download");
    return host;
}

int check_shape(const ShapeUnderTest& shape) {
    const auto packed = shape.generator(shape.n, shape.k, 7U);
    DeviceBuffer payload(packed.payload.size());
    payload.copy_from_host(packed.payload.data(), payload.bytes);
    const Weight weight = packed.device_weight(payload.p);
    const auto activation = random_activation(shape.k, kWidest, 11U);
    DeviceBuffer x_all(activation.size() * sizeof(std::uint16_t));
    x_all.copy_from_host(activation.data(), x_all.bytes);

    const auto reference = run(shape, weight, x_all, kWidest);
    int failures = 0;
    for (const std::int32_t t : kTokens) {
        const auto got = run(shape, weight, x_all, t);
        std::int64_t differing = 0;
        std::int32_t first     = -1;
        for (std::int32_t column = 0; column < t; ++column) {
            const auto* a = got.data() + static_cast<std::size_t>(column) * shape.n;
            const auto* b = reference.data() + static_cast<std::size_t>(column) * shape.n;
            if (std::memcmp(a, b, static_cast<std::size_t>(shape.n) * 2) != 0) {
                ++differing;
                if (first < 0) { first = column; }
            }
        }
        if (differing != 0) {
            ++failures;
            std::cout << "DIFFERS " << shape.name << " T=" << t << " vs T=" << kWidest << ": "
                      << differing << " of " << t << " columns, first " << first << '\n';
        }
    }
    for (const Window& window : kWindows) {
        const auto got         = run(shape, weight, x_all, window.tokens, window.offset);
        std::int64_t differing = 0;
        std::int32_t first     = -1;
        for (std::int32_t column = 0; column < window.tokens; ++column) {
            const auto* a = got.data() + static_cast<std::size_t>(column) * shape.n;
            const auto* b =
                reference.data() + static_cast<std::size_t>(window.offset + column) * shape.n;
            if (std::memcmp(a, b, static_cast<std::size_t>(shape.n) * 2) != 0) {
                ++differing;
                if (first < 0) { first = column; }
            }
        }
        if (differing != 0) {
            ++failures;
            std::cout << "DIFFERS " << shape.name << " columns [" << window.offset << ", "
                      << window.offset + window.tokens
                      << ") as their own call vs inside T=" << kWidest << ": " << differing
                      << " of " << window.tokens << " columns, first " << first << '\n';
        }
    }
    std::cout << (failures == 0 ? "invariant " : "VARIANT ") << shape.name << '\n';
    return failures;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const ShapeUnderTest shapes[] = {
        // 26B-A4B
        {"26B q N8192K2816", make_nvfp4_weight, 8192, 2816, ops::LinearPolicy::AllowA4},
        {"26B k N1024K2816", make_nvfp4_weight, 1024, 2816, ops::LinearPolicy::AllowA4},
        {"26B o N2816K4096", make_nvfp4_weight, 2816, 4096, ops::LinearPolicy::AllowA4},
        {"26B o N2816K8192", make_nvfp4_weight, 2816, 8192, ops::LinearPolicy::AllowA4},
        {"26B down N2816K2112", make_nvfp4_weight, 2816, 2112, ops::LinearPolicy::AllowA4},
        {"26B gate_up Q8 N4224K2816 A16", make_q8_g32_fp16_weight, 4224, 2816,
         ops::LinearPolicy::A16Only},
        {"26B gate_up Q8 N4224K2816 A8", make_q8_g32_fp16_weight, 4224, 2816,
         ops::LinearPolicy::AllowA8},
        // The A16 profiles (--a16, --a16-full) run the same NVFP4 planes with BF16 activations.
        {"26B q N8192K2816 A16", make_nvfp4_weight, 8192, 2816, ops::LinearPolicy::A16Only},
        {"26B k N1024K2816 A16", make_nvfp4_weight, 1024, 2816, ops::LinearPolicy::A16Only},
        {"26B o N2816K4096 A16", make_nvfp4_weight, 2816, 4096, ops::LinearPolicy::A16Only},
        {"26B o N2816K8192 A16", make_nvfp4_weight, 2816, 8192, ops::LinearPolicy::A16Only},
        {"26B down N2816K2112 A16", make_nvfp4_weight, 2816, 2112, ops::LinearPolicy::A16Only},
        // 31B
        {"31B N16384K5376", make_nvfp4_weight, 16384, 5376, ops::LinearPolicy::AllowA4},
        {"31B N18432K5376", make_nvfp4_weight, 18432, 5376, ops::LinearPolicy::AllowA4},
        {"31B N2048K5376", make_nvfp4_weight, 2048, 5376, ops::LinearPolicy::AllowA4},
        {"31B N43008K5376", make_nvfp4_weight, 43008, 5376, ops::LinearPolicy::AllowA4},
        {"31B N5376K8192", make_nvfp4_weight, 5376, 8192, ops::LinearPolicy::AllowA4},
        {"31B N5376K16384", make_nvfp4_weight, 5376, 16384, ops::LinearPolicy::AllowA4},
        {"31B N5376K21504", make_nvfp4_weight, 5376, 21504, ops::LinearPolicy::AllowA4},
        {"31B N16384K5376 A16", make_nvfp4_weight, 16384, 5376, ops::LinearPolicy::A16Only},
        {"31B N18432K5376 A16", make_nvfp4_weight, 18432, 5376, ops::LinearPolicy::A16Only},
        {"31B N2048K5376 A16", make_nvfp4_weight, 2048, 5376, ops::LinearPolicy::A16Only},
        {"31B N43008K5376 A16", make_nvfp4_weight, 43008, 5376, ops::LinearPolicy::A16Only},
        {"31B N5376K8192 A16", make_nvfp4_weight, 5376, 8192, ops::LinearPolicy::A16Only},
        {"31B N5376K16384 A16", make_nvfp4_weight, 5376, 16384, ops::LinearPolicy::A16Only},
        {"31B N5376K21504 A16", make_nvfp4_weight, 5376, 21504, ops::LinearPolicy::A16Only},
    };
    // The prefill route profile is what the model runs its prefill calls under.
    const ops::ScopedPrefillRoutes prefill_routes;
    int failures = 0;
    try {
        for (const auto& shape : shapes) {
            if (generator_runs(shape.generator, shape.name)) { failures += check_shape(shape); }
        }
    } catch (const std::exception& error) {
        std::cerr << "Gemma 4 Linear token invariance: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma 4 Linear token invariance\n";
    return failures == 0 ? 0 : 1;
}
